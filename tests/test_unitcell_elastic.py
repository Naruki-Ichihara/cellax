import cellax as cell
import os
import jax
import jax.numpy as np
from cellax.fem.problem import Problem, DirichletBC
from cellax.fem.generate_mesh import rectangle_mesh, Mesh, get_meshio_cell_type
from cellax.pbc import PeriodicPairing, prolongation_matrix
from cellax.fem.solver import solver, ad_wrapper
from cellax.fem.utils import save_sol

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

data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 100
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

left = unitcell.edge_function([0], [0], excluding_corner=False)
right = unitcell.edge_function([0], [1], excluding_corner=False)
map_lr = unitcell.mapping(left, right)

bottom = unitcell.edge_function([1], [0], excluding_corner=False)
top = unitcell.edge_function([1], [1], excluding_corner=False)

dirichlet_val = lambda x: 0
dirichlet_bc_bottom_x = DirichletBC(bottom, 0, lambda x: 0)
dirichlet_bc_bottom_y = DirichletBC(bottom, 1, lambda x: 0)
dirichlet_bc_top_x = DirichletBC(top, 0, lambda x: 0)
dirichlet_bc_top_y = DirichletBC(top, 1, lambda x: 0.1)
bcs = [dirichlet_bc_bottom_y, dirichlet_bc_top_y]

neumann_bc_subdomains = None

pair = PeriodicPairing(left, right, map_lr, 0)
P = prolongation_matrix([pair], unitcell.mesh, vec, 0)

problem = LinearElasticity(unitcell.mesh, vec=vec, dim=dim, ele_type=ele_type, 
                           dirichlet_bcs=bcs, prolongation_matrix=P, neumann_subdomains=neumann_bc_subdomains)

sol_list = solver(problem, solver_options={'jax_solver': {}})
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)