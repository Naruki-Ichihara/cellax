import jax
import jax.numpy as np
from cellax.fem.problem import Problem
from typing import Optional

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
            E_r = E_min + (E - E_min) * rho **3
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


