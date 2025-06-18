import cellax as cell
import os
import itertools
import jax
import jax.numpy as np
from cellax.fem.problem import Problem, DirichletBC
from cellax.fem.generate_mesh import box_mesh, Mesh, get_meshio_cell_type
from cellax.pbc import PeriodicPairing, prolongation_matrix, periodic_bc_3D
import cellax.tpms as tpms
from cellax.fem.solver import solver, ad_wrapper
from cellax.utils import save_as_vtk
from cellax.graphs import sc, bcc, fcc

# Material parameters.
E = 70e3
E_min = 1e-8
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Weak forms.
class LinearElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E_reduced = E_min + (E - E_min)*rho**5
            mu = E_reduced/(2.*(1.+nu))
            lmbda = E_reduced*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress
    def set_params(self, params):
        if len(params) != self.fes[0].num_total_nodes:
            raise ValueError(f"Number of parameters {len(params)} does not match number of nodes {self.fes[0].num_total_nodes}.")

        self.full_params = params

        # nodal to quad interpolation
        rho_cells = params[self.fes[0].cells]  # (num_cells, num_nodes)
        shape_vals = self.fes[0].shape_vals   # (num_quads, num_nodes)
        rho_quads = np.einsum("qn,cn->cq", shape_vals, rho_cells)  # (num_cells, num_quads)

        self.internal_vars = [rho_quads]
# Mesh
data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 10
L = 1.0
vec = 3
dim = 3
ele_type = "HEX8"
class Unitcell_3D(cell.Unitcell):
    def construct_mesh(self):
        cell_type = get_meshio_cell_type(ele_type)
        meshio_mesh = box_mesh(N, N, N, L, L, L)
        mesh = Mesh(meshio_mesh.points[:, :3], meshio_mesh.cells_dict[cell_type])
        return mesh
    
unitcell = Unitcell_3D()

# Perturbation
exx = 1.0
exy = 0.0
exz = 0.0

eyy = 0.0
ezy = 0.0
ezz = 0.0

E_macro = np.array([[exx, exy, exz],
                    [exy, eyy, ezy],
                    [exz, ezy, ezz]])

P = prolongation_matrix(periodic_bc_3D(unitcell, 3), 
                           unitcell.mesh, vec, 0)
P_rho = prolongation_matrix(periodic_bc_3D(unitcell, 1),
                           unitcell.mesh, 1, 0)

macro_disp = unitcell.macro_disp(E_macro)


# Construct the problem.
problem = LinearElasticity(unitcell.mesh, vec=vec, dim=dim, ele_type=ele_type, 
                           prolongation_matrix=P, 
                           macro_term=macro_disp)

# Solve
fwd_pred = ad_wrapper(problem, solver_options={'jax_solver': {}}, adjoint_solver_options={'jax_solver': {}})
#rho = jax.vmap(tpms.schoen_gyroid, in_axes=(0, None, None))(unitcell.cell_centers, [1, 1, 1], [0., 0., 0.])
#rho = jax.vmap(lambda x: 1)(unitcell.cell_centers)  # Use a constant density for testing
#rho = jax.vmap(bcc(radius=0.2, scale=L), in_axes=(0,))(unitcell.points)
rho_reduced = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=0.5, maxval=1.0)
rho = P_rho @ rho_reduced
sol_list  = fwd_pred(rho)
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')

from cellax.utils import compute_macro_stress_linear_elastic

u_sol = sol_list[0]  # (num_dofs,)
sigma_avg = compute_macro_stress_linear_elastic(problem, E, nu, u_sol, rho=rho, E_min=E_min)
print(f"sigma_avg: {sigma_avg}")


u_grad = problem.fes[0].sol_to_grad(u_sol)  # (num_cells, num_quads, dim, dim)
strain_field = jax.vmap(lambda u_grad: 0.5 * (u_grad + u_grad.T))(u_grad.reshape(-1, 3, 3))  # (num_cells*num_quads, dim, dim)
strain_field = strain_field.reshape(u_grad.shape)  # (num_cells, num_quads, dim, dim)
strain_cells = np.mean(strain_field, axis=1)  # Average over quadrature points
save_as_vtk(problem.fe, vtk_path, cell_infos=[('strain', strain_cells)], point_infos=[('rho', problem.full_params[:]), ('displacement', u_sol)])