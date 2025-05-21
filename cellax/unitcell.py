import jax
import jax.numpy as np
import numpy as onp
from cellax.fem.problem import Problem
from cellax.fem.generate_mesh import Mesh
from itertools import product

from abc import ABC, abstractmethod
from typing import Any, Tuple
from cellax.fem import logger

class Unitcell(ABC):
    """Represents a unit cell in the computational domain."""

    def __init__(self, atol: float=1e-5, **kwargs: Any):
        self.mesh = self.construct_mesh(**kwargs)
        self.points = self.mesh.points
        self.atol = atol

    @abstractmethod
    def construct_mesh(self, **kwargs: Any) -> Mesh:
        """Constructs the mesh for the unit cell."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the boundary of the mesh.
        Returns:
            np.ndarray: The boundary of the mesh.
                [(minimum point), (maximum point)]
        """
        mesh = self.mesh
        # Get the boundary of the mesh
        return np.min(self.points, axis=0), np.max(self.points, axis=0)
    
    @property
    def corners(self) -> np.ndarray:
        """
        Get the corners of the unit cell in N-dimensional space.
    
        Returns:
            np.ndarray: Array of shape (2^N, N), listing all corners of the bounding box.
        """
        min_corner, max_corner = self.bounds
        dim = min_corner.shape[0]
        corner_list = []
        for bits in product([0, 1], repeat=dim):
            corner = np.where(np.array(bits), max_corner, min_corner)
            corner_list.append(corner)

        return np.array(corner_list)
    
    @property
    def corner_mask(self) -> np.ndarray:
        """Boolean mask for which mesh points are corners of the unit cell.

        Returns:
            np.ndarray: Boolean array of shape (num_points,).
                True if the point is a corner, False otherwise.
        """
        condition = np.any(
        np.all(
                np.isclose(self.points[:, None, :], self.corners[None, :, :], atol=self.atol),
                axis=2
            ),
            axis=1
        )
        count_of_hit_corners = np.sum(condition)
        logger.debug(f"In corner_mask, Number of hit points: {count_of_hit_corners}")
        if count_of_hit_corners == 0:
            logger.warning("No hit points found in corner_mask.")
        return condition