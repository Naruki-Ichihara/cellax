import jax
import jax.numpy as np
import numpy as onp
from cellax.fem.problem import Problem
from cellax.fem.generate_mesh import Mesh
from itertools import product

from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable, Callable, Optional
from cellax.fem import logger

class Unitcell(ABC):
    """Represents a unit cell in the computational domain."""

    def __init__(self, atol: float=1e-5, **kwargs: Any):
        self.mesh = self.construct_mesh(**kwargs)
        self.points = self.mesh.points
        self.atol = atol
        self.lb, self.ub = self.bounds
        
        # Check masks
        corner_hits = np.sum(self.corner_mask)
        edge_hits = np.sum(self.edge_mask)
        face_hits = np.sum(self.face_mask)
        logger.debug(f"Number of points: {self.points.shape[0]}")
        logger.debug(f"Unitcell initialized with {corner_hits} corner points, {edge_hits} edge points, and {face_hits} face points.")

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
        # Get the boundary of the mesh
        return np.min(self.points, axis=0), np.max(self.points, axis=0)
    
    @property
    def corners(self) -> np.ndarray:
        """
        Get the corners of the unit cell in N-dimensional space.
    
        Returns:
            np.ndarray: Array of shape (2^N, N), listing all corners of the bounding box.
        """
        min_corner, max_corner = self.lb, self.ub
        dim = min_corner.shape[0]
        corner_list = []
        for bits in product([0, 1], repeat=dim):
            corner = np.where(np.array(bits), max_corner, min_corner)
            corner_list.append(corner)

        return np.array(corner_list)
    
    def perturbation(self, mean_gradient: np.ndarray, origin: Optional[np.ndarray]=None) -> np.ndarray:
        """Apply a perturbation to the unit cell points based on the mean gradient.

        Args:
            mean_gradient (np.ndarray): The mean gradient to apply.
            origin (Optional[np.ndarray]): The origin point for perturbation.

        Returns:
            np.ndarray: Perturbed points.
        """
        if origin is None:
            origin = np.min(self.points, axis=0)  # default to lower-left corner
        x_shifted = self.points - origin
        return x_shifted @ mean_gradient.T
    
    def is_corner(self, point: np.ndarray) -> bool:
        """Check if a point is a corner of the unit cell.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is a corner, False otherwise.
        """
        close_0 = np.isclose(point, self.lb, atol=self.atol)
        close_1 = np.isclose(point, self.ub, atol=self.atol)
        count_of_hit_corners = np.sum(np.logical_or(close_0, close_1))
        return np.equal(count_of_hit_corners, point.shape[0])
    
    def is_edge(self, point: np.ndarray) -> bool:
        """Check if a point is an edge of the unit cell.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is an edge, False otherwise.
        """
        close_0 = np.isclose(point, self.lb, atol=self.atol)
        close_1 = np.isclose(point, self.ub, atol=self.atol)
        boundary_hits = np.sum(np.logical_or(close_0, close_1))
        return np.logical_and(np.equal(boundary_hits, point.shape[0]-1), np.logical_not(self.is_corner(point)))
    
    def is_face(self, point: np.ndarray) -> bool:
        """Check if a point is a face of the unit cell.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is a face, False otherwise.
        """
        close_0 = np.isclose(point, self.lb, atol=self.atol)
        close_1 = np.isclose(point, self.ub, atol=self.atol)
        boundary_hits = np.sum(np.logical_or(close_0, close_1))
        return np.logical_and(np.equal(boundary_hits, point.shape[0]-2), 
                              np.logical_and(np.logical_not(self.is_edge(point)), np.logical_not(self.is_corner(point))))

    @property
    def corner_mask(self) -> np.ndarray:
        is_corner_vec = jax.vmap(self.is_corner)
        result = is_corner_vec(self.points)
        logger.debug(f"In corner_mask: number of corner points = {np.sum(result)}")
        return result

    @property
    def edge_mask(self) -> np.ndarray:
        is_edge_vec = jax.vmap(self.is_edge)
        result = is_edge_vec(self.points)
        logger.debug(f"In edge_mask: number of edge points = {np.sum(result)}")
        return result
    
    @property
    def face_mask(self) -> np.ndarray:
        is_face_vec = jax.vmap(self.is_face)
        result = is_face_vec(self.points)
        logger.debug(f"In face_mask: number of face points = {np.sum(result)}")
        return result
    
    def face_function(self, axis: int, value: float, excluding_edge=False, excluding_corner=False) -> Callable:
        """Get the mask for a specific face of the unit cell.

        Args:
            axis (int): The axis along which to check (0, 1, ..., N-1).
            value (float): The value of the coordinate along that axis.
            excluding_edge (bool): If True, exclude edges from the face mask.
            excluding_corner (bool): If True, exclude corners from the face mask.

        Returns:
            Callable: A function that checks if a point is on the specified face.
        """
        def fn(point: np.ndarray) -> bool:
            if excluding_corner:
                conds = [
                    np.isclose(point[axis], value, atol=self.atol),
                    np.logical_not(self.is_corner(point))
                ]
            elif excluding_edge:
                conds = [
                    np.isclose(point[axis], value, atol=self.atol),
                    np.logical_not(self.is_edge(point)),
                    np.logical_not(self.is_corner(point))
                ]
            else:
                conds = np.isclose(point[axis], value, atol=self.atol)
            return np.all(np.stack(conds), axis=0)
        return fn
    
    def edge_function(self, axes: Iterable[int], values: Iterable[float], excluding_corner=False) -> Callable:
        """Get the mask for a specific edge of the unit cell.

        Args:
            axes (Iterable): The axes along which to check (0, 1, ..., N-1).
            values (Iterable): The values of the coordinates along those axes.
            excluding_corner (bool): If True, exclude corners from the edge mask.
        Returns:
            Callable: A function that checks if a point is on the specified edge.
        """
        def fn(point: np.ndarray) -> bool:
            cond = np.ones((), dtype=bool)
            for axis, value in zip(axes, values):
                cond = np.logical_and(cond, np.isclose(point[axis], value, atol=self.atol))
            if excluding_corner:
                # Exclude corners from the edge mask
                return np.logical_and(cond, np.logical_not(self.is_corner(point)))
            else:
                return cond
        return fn
    
    def corner_function(self, values: Iterable[float]) -> Callable:
        """Get the mask for a specific corner of the unit cell.

        Args:
            values (Iterable): The values of the coordinates at the corner.

        Returns:
            Callable: A function that checks if a point is at the specified corner.
        """
        values = np.array(values)
        def fn(point: np.ndarray) -> bool:
            return np.all(np.isclose(point, values, atol=self.atol))
        return fn
    
    def mapping(self, master: Callable, slave: Callable) -> Callable:
        """Get the mapping function from master face to slave face.

        Args:
            master (Callable): Boolean filter function for master points.
            slave (Callable): Boolean filter function for slave points.

        Returns:
            Callable: Function that maps a point on the master face to the slave face.
        """
        master_mask = jax.vmap(master)(self.points)
        slave_mask = jax.vmap(slave)(self.points)

        master_pts = self.points[master_mask]
        slave_pts = self.points[slave_mask]

        if master_pts.shape[0] != slave_pts.shape[0]:
            raise ValueError("Master and slave point sets must have the same number of points.")

        deltas = slave_pts - master_pts

        def fn(point: np.ndarray) -> np.ndarray:
            dists = np.linalg.norm(master_pts - point, axis=1)
            idx = np.argmin(dists)
            return point + deltas[idx]

        return fn
            