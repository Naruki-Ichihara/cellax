import cellax as cell
import os
import jax
import jax.numpy as np
from cellax.fem.problem import Problem, DirichletBC
from cellax.fem.generate_mesh import rectangle_mesh, Mesh, get_meshio_cell_type
from cellax.pbc import PeriodicPairing, prolongation_matrix
from cellax.fem.solver import solver, ad_wrapper
from cellax.fem.utils import save_sol

# Material parameters.
E = 70e3
nu = 0.3
mu = E/(2.*(1.+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

# Weak forms.
class LinearElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def get_tensor_map(self):
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

# Mesh
data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 10
L = 1.0
vec = 2
dim = 2
ele_type = "QUAD4"
class Unitcell_2D(cell.Unitcell):
    def construct_mesh(self):
        cell_type = get_meshio_cell_type(ele_type)
        meshio_mesh = rectangle_mesh(N, N, L, L)
        mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict[cell_type])
        return mesh
    
unitcell = Unitcell_2D()

# Define Subdomains
left = unitcell.edge_function([0], [0], excluding_corner=True)
right = unitcell.edge_function([0], [1], excluding_corner=True)
bottom = unitcell.edge_function([1], [0], excluding_corner=True)
top = unitcell.edge_function([1], [1], excluding_corner=True)
corner_lb = unitcell.corner_function([0, 0])
corner_lt = unitcell.corner_function([0, 1])
corner_rb = unitcell.corner_function([1, 0])
corner_rt = unitcell.corner_function([1, 1])

# Define Mappings
map_lr = unitcell.mapping(left, right)
map_bt = unitcell.mapping(bottom, top)
map_lb_lt = unitcell.mapping(corner_lb, corner_lt)
map_lb_rt = unitcell.mapping(corner_lb, corner_rt)
map_lb_rb = unitcell.mapping(corner_lb, corner_rb)

# Edge pairings.
pairs_edge = [PeriodicPairing(left, right, map_lr, 0),
              PeriodicPairing(bottom, top, map_bt, 0)]
# Corner pairings.
pairs_corner = [PeriodicPairing(corner_lb, corner_lt, map_lb_lt, 0),
                PeriodicPairing(corner_lb, corner_rt, map_lb_rt, 0),
                PeriodicPairing(corner_lb, corner_rb, map_lb_rb, 0)]

P = prolongation_matrix(pairs_edge+pairs_corner, unitcell.mesh, vec, 0)

# Perturbation
exx = 0.
exy = 1.
eyy = 0.

E_macro = np.array([[exx, exy],
                    [exy, eyy]])
perturbation = unitcell.perturbation(E_macro)

# Construct the problem.
problem = LinearElasticity(unitcell.mesh, vec=vec, dim=dim, ele_type=ele_type, 
                           prolongation_matrix=P, 
                           perturbation=perturbation)

# Solve
sol_list = solver(problem, solver_options={'jax_solver': {}})
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)

u_sol = sol_list[0]
u_grad = problem.fes[0].sol_to_grad(u_sol)

def stress_fn(u_grad):
    epsilon = 0.5 * (u_grad + u_grad.transpose(0,2,1))
    sigma = lmbda * np.trace(epsilon, axis1=1, axis2=2)[:,None,None] * np.eye(problem.dim) + 2*mu*epsilon
    return sigma

stress_field = jax.vmap(stress_fn)(u_grad)

JxW = problem.JxW[:,0,:]
vol_total = np.sum(JxW)

stress_weighted = stress_field * JxW[:, :, None, None]
sigma_avg = np.sum(stress_weighted, axis=(0, 1)) / vol_total
print(sigma_avg)