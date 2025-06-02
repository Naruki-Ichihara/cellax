import jax
import jax.numpy as np
from typing import Optional, Callable, List

def schoen_gyroid(x: np.ndarray, cell_sizes: List[float], origin: Optional[List[float]] = [0, 0, 0], sharpeness: Optional[float] = 10.0) -> Callable:
    """Schoen gyroid surface function."""
    w_x = 1 / cell_sizes[0] * 2 * np.pi
    w_y = 1 / cell_sizes[1] * 2 * np.pi
    w_z = 1 / cell_sizes[2] * 2 * np.pi

    X, Y, Z = x[0], x[1], x[2]

    F = (np.cos(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) + np.cos(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) + np.cos(w_z * (Z + origin[2])) * np.sin(w_x * (X + origin[0])))
    rho = 0.5 + (1 / np.pi) * np.arctan(sharpeness * F)

    return rho

def schwarz_diamond(x: np.ndarray, cell_sizes: List[float], origin: Optional[List[float]] = [0, 0, 0], sharpeness: Optional[float] = 10.0) -> Callable:
    """Schwarz diamond surface function."""
    w_x = 1 / cell_sizes[0] * 2 * np.pi
    w_y = 1 / cell_sizes[1] * 2 * np.pi
    w_z = 1 / cell_sizes[2] * 2 * np.pi

    X, Y, Z = x[0], x[1], x[2]

    F = (np.cos(w_x * (X + origin[0])) * np.cos(w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.sin(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])))
    rho = 0.5 + (1 / np.pi) * np.arctan(sharpeness * F)

    return rho

def body_diagonals(x: np.ndarray, cell_sizes: List[float], origin: Optional[List[float]] = [0, 0, 0], sharpeness: Optional[float] = 10.0) -> Callable:
    """Schwarz diamond surface function."""
    w_x = 1 / cell_sizes[0] * 2 * np.pi
    w_y = 1 / cell_sizes[1] * 2 * np.pi
    w_z = 1 / cell_sizes[2] * 2 * np.pi

    X, Y, Z = x[0], x[1], x[2]

    F = (2 * (np.cos(w_x * ((X + origin[0]))) * np.cos(w_y * (Y + origin[1])) + np.cos(w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.cos(w_z * (Z + origin[2])) * np.cos(w_x * (X + origin[0]))) - (np.cos(2 * w_x * (X + origin[0])) + np.cos(2 * w_y * (Y + origin[1])) + np.cos(2 * w_z * (Z + origin[2])))) # O
    rho = 0.5 + (1 / np.pi) * np.arctan(sharpeness * F)

    return rho