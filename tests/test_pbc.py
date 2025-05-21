import cellax as cell
import os
import jax
import jax.numpy as np
import gmsh
import meshio
from cellax.fem.problem import Problem
from cellax.fem.generate_mesh import Mesh, get_meshio_cell_type
from cellax.fem.basis import get_elements

def gmsh_mesh(data_dir, degree):
    """
    Generate a mesh
    Reference:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_8_3/tutorial/python/t1.py
    """
    msh_dir = os.path.join(data_dir, f'msh')
    os.makedirs(msh_dir, exist_ok=True)
    file_path = os.path.join(msh_dir, f't1.msh')

    gmsh.initialize()
    gmsh.model.add("t1")
    lc = 1e-2
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1., 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1., 1., 0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0, 1., 0, lc)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, p4, 3)
    gmsh.model.geo.addLine(4, 1, p4)
    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(file_path)

    return file_path

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
ele_type = 'TRI6'
_, _, _, _, degree, _ = get_elements(ele_type)
msh_file_path = gmsh_mesh(data_dir, degree)
cell_type = get_meshio_cell_type(ele_type)
meshio_mesh = meshio.read(msh_file_path)
mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict[cell_type])
vec = 1
dim = 2

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1., atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], 1., atol=1e-5)

def dirichlet_val(point):
    return 0.

def mapping_x(point_A):
    point_B = point_A + np.array([1., 0])
    return point_B

def mapping_y(point_A):
    point_B = point_A + np.array([0, 1.])
    return point_B

pair_LR = cell.PeriodicPairing(
    left, right, mapping_x, 0
)
pair_BT = cell.PeriodicPairing(
    bottom, top, mapping_y, 0
)

P = cell.prolongation_matrix([pair_LR], mesh, 1)
print(P.shape)