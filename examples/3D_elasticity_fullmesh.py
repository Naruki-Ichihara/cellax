import cellax as cell
import os
import itertools
import jax
import jax.numpy as np
from cellax.fem.problem import Problem, DirichletBC
from cellax.fem.generate_mesh import box_mesh, Mesh, get_meshio_cell_type
from cellax.pbc import PeriodicPairing, prolongation_matrix
import cellax.tpms as tpms
from cellax.fem.solver import solver, ad_wrapper
from cellax.fem.utils import save_sol2
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
        # length of params should be equal to the number of elements in the mesh.
        self.full_params = params
        if len(params) != self.fes[0].num_cells:
            raise ValueError(f"Number of parameters {len(params)} does not match number of cells {self.fes[0].num_cells}.")
        # Expand the parameters to each integration point.
        rhos = np.repeat(params[:, None], self.fes[0].num_quads, axis=1)
        self.internal_vars = [rhos]
# Mesh
data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 80
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

face_pairs = []
for axis in range(dim):
    master_fn = unitcell.face_function(axis, 0, excluding_edge=True)
    slave_fn = unitcell.face_function(axis, L, excluding_edge=True)
    for i in range(vec):
        face_pairs.append(PeriodicPairing(
            master_fn,
            slave_fn,
            unitcell.mapping(master_fn, slave_fn), [i]))

edge_pairs = []
for axes in [[1, 2], [0, 2], [0, 1]]:
    for values in [[L, 0], [L, L], [0, L]]:
        master_fn = unitcell.edge_function(axes, [0, 0], excluding_corner=True)
        slave_fn = unitcell.edge_function(axes, values, excluding_corner=True)
        for i in range(vec):
            edge_pairs.append(
                PeriodicPairing(
                    master_fn,
                    slave_fn,
                    unitcell.mapping(master_fn, slave_fn), [i]))

corner_origin = [0, 0, 0]
corner_pairs = []
for corner in itertools.product([0, L], repeat=dim):
    if corner == (0, 0, 0):
        continue
    master_fn = unitcell.corner_function(corner_origin)
    slave_fn = unitcell.corner_function(corner)
    for i in range(vec):
        corner_pairs.append(
            PeriodicPairing(
                master_fn,
                slave_fn,
                unitcell.mapping(master_fn, slave_fn), [i]))

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

P = prolongation_matrix(corner_pairs + edge_pairs + face_pairs, 
                           unitcell.mesh, vec, 0)

u_affine = unitcell.u_affine(E_macro)


# Construct the problem.
problem = LinearElasticity(unitcell.mesh, vec=vec, dim=dim, ele_type=ele_type, 
                           prolongation_matrix=P, 
                           u_affine=u_affine)

# Solve
fwd_pred = ad_wrapper(problem, solver_options={'jax_solver': {}}, adjoint_solver_options={'jax_solver': {}})
#rho = jax.vmap(tpms.schoen_gyroid, in_axes=(0, None, None))(unitcell.cell_centers, [1, 1, 1], [0., 0., 0.])
#rho = jax.vmap(lambda x: 1)(unitcell.cell_centers)  # Use a constant density for testing
rho = jax.vmap(bcc(thickness=0.1, scale=L), in_axes=(0,))(
    unitcell.cell_centers
)
sol_list  = fwd_pred(rho)
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')

from cellax.utils import compute_macro_stress_linear_elastic

u_sol = sol_list[0]  # (num_dofs,)
sigma_avg = compute_macro_stress_linear_elastic(problem, E, nu, u_sol, rho=rho, E_min=E_min)
print(f"sigma_avg: {sigma_avg}")

# Backward
def J(rho):
    u_sol = sol_list[0]  # (num_dofs,)
    sigma_avg = compute_macro_stress_linear_elastic(problem, E, nu, u_sol, rho=rho, E_min=E_min)
    return np.sum(sigma_avg)

def strain_fn(u_grad):
    """Compute the strain tensor from the displacement gradient."""
    return 0.5 * (u_grad + u_grad.T)  # Symmetric strain tensor

u_grad = problem.fes[0].sol_to_grad(u_sol)  # (num_cells, num_quads, dim, dim)
strain_field = jax.vmap(strain_fn)(u_grad.reshape(-1, 3, 3))  # (num_cells*num_quads, dim, dim)
strain_field = strain_field.reshape(u_grad.shape)  # (num_cells, num_quads, dim, dim)
strain_cells = np.mean(strain_field, axis=1)  # Average over quadrature points
save_sol2(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('rho', problem.full_params[:]), ('strain', strain_cells)])