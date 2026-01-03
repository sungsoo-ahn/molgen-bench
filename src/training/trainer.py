"""Training loop for molecular generative models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict
from tqdm import tqdm
import time
from pathlib import Path

from .scaling_laws import ScalingLawTracker


class Trainer:
    """Trainer for molecular generative models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        track_scaling: bool = True
    ):
        """Initialize trainer.

        Args:
            model: Generative model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            mixed_precision: Whether to use mixed precision training
            gradient_clip: Gradient clipping value
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            checkpoint_dir: Directory to save checkpoints
            track_scaling: Whether to track scaling laws
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Model size (always compute for logging)
        self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Scaling law tracking
        self.track_scaling = track_scaling
        if track_scaling:
            self.scaling_tracker = ScalingLawTracker(
                save_dir=str(self.checkpoint_dir / "scaling_logs")
            )
        else:
            self.scaling_tracker = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            dict with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            # Move to device
            batch = batch.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with autocast():
                    loss, info = self.model(
                        positions=batch.positions,
                        atom_types=batch.atom_types,
                        edge_index=batch.edge_index,
                        batch_idx=batch.batch_idx
                    )
            else:
                loss, info = self.model(
                    positions=batch.positions,
                    atom_types=batch.atom_types,
                    edge_index=batch.edge_index,
                    batch_idx=batch.batch_idx
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

            # Periodic logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"Step {self.global_step}: train_loss={avg_loss:.4f}")

            # Periodic evaluation
            if self.global_step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                print(f"Step {self.global_step}: val_loss={val_metrics['loss']:.4f}")

                # Save checkpoint if best
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best_model.pt")

                # Track scaling
                if self.track_scaling:
                    self.scaling_tracker.log(
                        step=self.global_step,
                        train_loss=avg_loss,
                        val_loss=val_metrics["loss"],
                        model_size=self.model_size,
                        dataset_size=len(self.train_loader.dataset),
                        wall_time=time.time() - start_time
                    )

                self.model.train()

        epoch_time = time.time() - start_time

        return {
            "loss": total_loss / num_batches,
            "time": epoch_time,
            "steps": num_batches
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set.

        Returns:
            dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            if self.mixed_precision:
                with autocast():
                    loss, info = self.model(
                        positions=batch.positions,
                        atom_types=batch.atom_types,
                        edge_index=batch.edge_index,
                        batch_idx=batch.batch_idx
                    )
            else:
                loss, info = self.model(
                    positions=batch.positions,
                    atom_types=batch.atom_types,
                    edge_index=batch.edge_index,
                    batch_idx=batch.batch_idx
                )

            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / num_batches}

    def train(self, num_epochs: int):
        """Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model size: {self.model_size:,} parameters")
        print(f"Training samples: {len(self.train_loader.dataset):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            metrics = self.train_epoch()
            print(f"\nEpoch {epoch} completed:")
            print(f"  Train loss: {metrics['loss']:.4f}")
            print(f"  Time: {metrics['time']:.2f}s")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Save final checkpoint
        self.save_checkpoint("final_model.pt")

        # Save scaling data
        if self.track_scaling:
            self.scaling_tracker.save()
            summary = self.scaling_tracker.get_summary()
            print("\n" + "="*50)
            print("Scaling Law Summary:")
            print("="*50)
            for key, value in summary.items():
                print(f"{key}: {value}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        filepath = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        filepath = self.checkpoint_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from {filepath}")
        print(f"  Epoch: {self.epoch}, Step: {self.global_step}")
