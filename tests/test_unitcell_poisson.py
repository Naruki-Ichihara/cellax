import cellax as cell
import os
import jax
import jax.numpy as np
from cellax.fem.problem import Problem, DirichletBC
from cellax.fem.generate_mesh import rectangle_mesh, Mesh, get_meshio_cell_type
from cellax.pbc import PeriodicPairing, prolongation_matrix
from cellax.fem.solver import solver, ad_wrapper
from cellax.fem.utils import save_sol

class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x, theta: x * theta

    def get_mass_map(self):
        def mass_map(u, x, theta):
            dx = x[0] - 0.5
            dy = x[1] - 0.5
            val = x[0]*np.sin(5.0*np.pi*x[1]) + 1.0*np.exp(-(dx*dx + dy*dy)/0.02)
            return np.array([-val])
        return mass_map

    def set_params(self, theta):
        thetas = theta * np.ones((self.fes[0].num_cells, self.fes[0].num_quads))
        self.internal_vars = [thetas]

data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 100
L = 1.0
vec = 1
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
dirichlet_bc_bottom = DirichletBC(bottom, 0, lambda x: 0)
dirichlet_bc_top = DirichletBC(top, 0, lambda x: 0)
bcs = [dirichlet_bc_bottom, dirichlet_bc_top]

pair = PeriodicPairing(left, right, map_lr, 0)
P = prolongation_matrix([pair], unitcell.mesh, vec, 0)

problem = Poisson(unitcell.mesh, vec=vec, dim=dim, ele_type=ele_type, dirichlet_bcs=bcs, prolongation_matrix=P)

fwd_pred = ad_wrapper(problem, solver_options={'jax_solver': {}}, adjoint_solver_options={'jax_solver': {}}) 
theta = 1.
sol_list = fwd_pred(theta)
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)