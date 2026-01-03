"""3D toy molecular datasets for testing generative models.

Each sample is a set of 3D points forming geometric shapes with atom type assignments.
This has the same feature structure as QM9 (positions + atom types) for proper testing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Literal, Optional, Dict, List


# Atom types matching QM9 (H, C, N, O, F)
ATOM_TYPES = [1, 6, 7, 8, 9]
ATOM_PROBS = [0.5, 0.3, 0.1, 0.08, 0.02]  # Biased towards H and C like real molecules


class ToyMolecular3DDataset(Dataset):
    """3D toy molecular dataset with geometric shapes.

    Each sample has the same structure as QM9:
    - positions: (N, 3) 3D coordinates
    - atom_types: (N,) atomic numbers from [1, 6, 7, 8, 9]
    - charges: (N,) atomic charges (zeros)

    Shapes available (2D patterns - easy to visually inspect):
    - circle: Points on a circle in XY plane
    - star: 5-pointed star in XY plane
    - grid: Regular grid pattern in XY plane

    Shapes available (3D patterns):
    - sphere: Points distributed on a 3D sphere surface
    - cube: Points on cube edges
    - spiral: 3D helix/spiral
    - cluster: Random 3D cluster
    - tetrahedron: Regular tetrahedron edges
    - mixed: Random mix of above
    """

    SHAPE_TYPES_2D = ["circle", "star", "grid"]
    SHAPE_TYPES_3D = ["sphere", "cube", "spiral", "cluster", "tetrahedron"]
    ALL_SHAPE_TYPES = SHAPE_TYPES_2D + SHAPE_TYPES_3D

    def __init__(
        self,
        shape_type: Literal["circle", "star", "grid", "sphere", "cube", "spiral", "cluster", "tetrahedron", "mixed"] = "circle",
        num_samples: int = 1000,
        num_atoms_range: tuple = (10, 30),
        noise: float = 0.02,
        scale: float = 1.0,
        seed: Optional[int] = 42
    ):
        """Initialize 3D toy molecular dataset.

        Args:
            shape_type: Type of 3D shapes to generate
            num_samples: Number of samples in dataset
            num_atoms_range: (min, max) number of atoms per molecule
            noise: Gaussian noise to add to positions
            scale: Scale of the shapes (in Angstrom-like units)
            seed: Random seed
        """
        self.shape_type = shape_type
        self.num_samples = num_samples
        self.num_atoms_range = num_atoms_range
        self.noise = noise
        self.scale = scale

        if seed is not None:
            np.random.seed(seed)

        # Pre-generate all samples
        self.samples = [self._generate_sample(i) for i in range(num_samples)]

    def _generate_circle(self, num_atoms: int) -> np.ndarray:
        """Generate points on a circle in XY plane (z=0)."""
        theta = np.linspace(0, 2 * np.pi, num_atoms, endpoint=False)
        x = np.cos(theta) * 0.5
        y = np.sin(theta) * 0.5
        z = np.zeros(num_atoms)
        points = np.stack([x, y, z], axis=1) * self.scale
        return points

    def _generate_star(self, num_atoms: int, num_points: int = 5) -> np.ndarray:
        """Generate points on a 5-pointed star in XY plane (z=0)."""
        # Alternate between outer and inner radius
        outer_radius = 0.5
        inner_radius = 0.2

        points = []
        # Generate star vertices (alternating outer/inner)
        num_vertices = num_points * 2
        for i in range(num_vertices):
            angle = i * np.pi / num_points - np.pi / 2  # Start from top
            radius = outer_radius if i % 2 == 0 else inner_radius
            points.append([radius * np.cos(angle), radius * np.sin(angle), 0])

        vertices = np.array(points)

        # Distribute atoms along the star edges
        result = []
        atoms_per_edge = num_atoms // num_vertices
        remainder = num_atoms % num_vertices

        for i in range(num_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % num_vertices]
            n_pts = atoms_per_edge + (1 if i < remainder else 0)
            if n_pts > 0:
                t = np.linspace(0, 1, n_pts, endpoint=False)
                for ti in t:
                    point = v1 * (1 - ti) + v2 * ti
                    result.append(point)

        result = np.array(result[:num_atoms]) * self.scale
        return result

    def _generate_grid(self, num_atoms: int) -> np.ndarray:
        """Generate points on a regular grid in XY plane (z=0)."""
        # Find grid dimensions closest to square
        grid_size = int(np.ceil(np.sqrt(num_atoms)))

        # Generate grid points
        x = np.linspace(-0.5, 0.5, grid_size)
        y = np.linspace(-0.5, 0.5, grid_size)
        xx, yy = np.meshgrid(x, y)

        points = np.stack([xx.flatten(), yy.flatten(), np.zeros(grid_size * grid_size)], axis=1)
        points = points[:num_atoms] * self.scale
        return points

    def _generate_sphere(self, num_atoms: int) -> np.ndarray:
        """Generate points on a 3D sphere surface."""
        # Use Fibonacci sphere for uniform distribution
        indices = np.arange(num_atoms, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_atoms)
        theta = np.pi * (1 + np.sqrt(5)) * indices

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points = np.stack([x, y, z], axis=1) * self.scale
        return points

    def _generate_cube(self, num_atoms: int) -> np.ndarray:
        """Generate points on cube edges."""
        points = []
        edges_per_atom = num_atoms // 12  # 12 edges in a cube

        # Define cube vertices
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 0.5

        # Define edges (pairs of vertex indices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
        ]

        # Distribute points along edges
        for i, (v1, v2) in enumerate(edges):
            n_pts = edges_per_atom + (1 if i < num_atoms % 12 else 0)
            if n_pts > 0:
                t = np.linspace(0, 1, n_pts, endpoint=False)
                for ti in t:
                    point = vertices[v1] * (1 - ti) + vertices[v2] * ti
                    points.append(point)

        points = np.array(points[:num_atoms]) * self.scale
        return points

    def _generate_spiral(self, num_atoms: int) -> np.ndarray:
        """Generate points along a 3D helix/spiral."""
        t = np.linspace(0, 4 * np.pi, num_atoms)
        r = 0.5  # radius

        x = r * np.cos(t)
        y = r * np.sin(t)
        z = t / (4 * np.pi) - 0.5  # height from -0.5 to 0.5

        points = np.stack([x, y, z], axis=1) * self.scale
        return points

    def _generate_cluster(self, num_atoms: int) -> np.ndarray:
        """Generate random 3D cluster of points."""
        # Random center
        center = np.random.randn(3) * 0.2

        # Points clustered around center
        points = np.random.randn(num_atoms, 3) * 0.3 + center
        return points * self.scale

    def _generate_tetrahedron(self, num_atoms: int) -> np.ndarray:
        """Generate points on tetrahedron edges."""
        # Regular tetrahedron vertices
        vertices = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]) / np.sqrt(3) * 0.5

        # 6 edges
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        points = []
        points_per_edge = num_atoms // 6

        for i, (v1, v2) in enumerate(edges):
            n_pts = points_per_edge + (1 if i < num_atoms % 6 else 0)
            if n_pts > 0:
                t = np.linspace(0, 1, n_pts, endpoint=False)
                for ti in t:
                    point = vertices[v1] * (1 - ti) + vertices[v2] * ti
                    points.append(point)

        points = np.array(points[:num_atoms]) * self.scale
        return points

    def _get_shape_generator(self, shape_name: str):
        """Get the generator function for a shape type."""
        generators = {
            "circle": self._generate_circle,
            "star": self._generate_star,
            "grid": self._generate_grid,
            "sphere": self._generate_sphere,
            "cube": self._generate_cube,
            "spiral": self._generate_spiral,
            "cluster": self._generate_cluster,
            "tetrahedron": self._generate_tetrahedron,
        }
        return generators.get(shape_name)

    def _generate_sample(self, idx: int) -> Dict:
        """Generate a single molecular sample."""
        # Random number of atoms
        num_atoms = np.random.randint(self.num_atoms_range[0], self.num_atoms_range[1] + 1)

        # Generate shape based on type
        if self.shape_type == "mixed":
            # Randomly choose from all shape types
            shape = np.random.choice(self.ALL_SHAPE_TYPES)
            generator = self._get_shape_generator(shape)
        else:
            generator = self._get_shape_generator(self.shape_type)

        if generator is None:
            raise ValueError(f"Unknown shape type: {self.shape_type}")

        positions = generator(num_atoms)

        # Add noise
        if self.noise > 0:
            positions += np.random.randn(*positions.shape) * self.noise

        # Assign random atom types (biased towards H and C like real molecules)
        atom_types = np.random.choice(ATOM_TYPES, size=num_atoms, p=ATOM_PROBS)

        # Charges are zeros (like QM9)
        charges = np.zeros(num_atoms)

        return {
            "positions": positions.astype(np.float32),
            "atom_types": atom_types.astype(np.int64),
            "charges": charges.astype(np.float32),
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        """Get a single molecular sample.

        Returns:
            Dict with:
                - positions: (N, 3) tensor of 3D coordinates
                - atom_types: (N,) tensor of atomic numbers
                - charges: (N,) tensor of charges (zeros)
        """
        sample = self.samples[idx]

        return {
            "positions": torch.from_numpy(sample["positions"]),
            "atom_types": torch.from_numpy(sample["atom_types"]),
            "charges": torch.from_numpy(sample["charges"]),
        }
