import jax
import jax.numpy as np
import numpy as onp
from cellax.fem.problem import Problem
from typing import Optional
import meshio
import os

from cellax.fem import logger
from cellax.fem.generate_mesh import get_meshio_cell_type

def save_as_vtk(fe, sol_file, cell_infos=None, point_infos=None):
    if cell_infos is None and point_infos is None:
        raise ValueError("At least one of cell_infos or point_infos must be provided.")
    cell_type = get_meshio_cell_type(fe.ele_type)
    sol_dir = os.path.dirname(sol_file)
    os.makedirs(sol_dir, exist_ok=True)

    out_mesh = meshio.Mesh(points=fe.points, cells={cell_type: fe.cells})

    if cell_infos is not None:
        out_mesh.cell_data = {}
        for cell_info in cell_infos:
            name, data = cell_info
            assert data.shape[0] == fe.num_cells, (
                f"cell data wrong shape, got {data.shape}, expected first dim = {fe.num_cells}"
            )
            data = onp.array(data, dtype=onp.float32)
            if data.ndim == 3:
                # Tensor (num_cells, 3, 3) -> flatten to (num_cells, 9)
                data = data.reshape(fe.num_cells, -1)
            elif data.ndim == 2:
                # Vector (num_cells, n) is OK
                pass
            else:
                # Scalar (num_cells,)
                data = data.reshape(fe.num_cells, 1)
            out_mesh.cell_data[name] = [data]

    if point_infos is not None:
        for point_info in point_infos:
            name, data = point_info
            out_mesh.point_data[name] = onp.array(data, dtype=onp.float32)

    out_mesh.write(sol_file)

def compute_macro_stress_linear_elastic(problem: Problem, E: float, nu: float, u_sol: np.ndarray, rho: Optional[np.ndarray] = None, E_min: float=1e-5) -> np.ndarray:
    """
    Compute the macroscopic stress tensor for a linear elasticity problem.

    Args:
        problem (Problem): The JAX-FEM problem instance.
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio of the material.
        u_sol (np.ndarray): Displacement solution array of shape (num_cells, num_quads, dim).
        rho (Optional[np.ndarray]): Density or volume fraction array of shape (num_cells, num_quads). If None, it will be computed from the internal variables.
        E_min (float): Minimum Young's modulus to avoid singularities.
        
    Returns:
        np.ndarray: The macroscopic stress tensor of shape (3, 3).
    """
    u_grad = problem.fes[0].sol_to_grad(u_sol)  # (num_cells, num_quads, dim, dim)
    rho_quads = problem.internal_vars[0]  # shape: (num_cells, num_quads)

    def stress_fn(u_grad, rho):
        if rho is not None:
            E_r = E_min + (E - E_min) * rho**5
        else:
            E_r = E
        mu_r = E_r / (2.0 * (1.0 + nu))
        lmbda_r = E_r * nu / ((1 + nu) * (1 - 2 * nu))
        epsilon = 0.5 * (u_grad + u_grad.T)
        sigma = lmbda_r * np.trace(epsilon) * np.eye(3) + 2 * mu_r * epsilon
        return sigma
    
    u_grad_flat = u_grad.reshape(-1, 3, 3)             # (num_cells*num_quads, 3, 3)
    rho_flat = rho_quads.reshape(-1)                   # (num_cells*num_quads,)
    stress_field = jax.vmap(stress_fn)(u_grad_flat, rho_flat)
    stress_field = stress_field.reshape(u_grad.shape)  # (num_cells, num_quads, 3, 3)
    JxW = problem.JxW[:, 0, :]
    vol_total = np.sum(JxW)
    stress_weighted = stress_field * JxW[:, :, None, None]  # (cells, quads, 3, 3)
    sigma_avg = np.sum(stress_weighted, axis=(0, 1)) / vol_total  # (3, 3)

    return sigma_avg


