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

ele_type = 'TRI6'

class Unitcell(cell.Unitcell):
    def construct_mesh(self, ele_type):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        _, _, _, _, degree, _ = get_elements(ele_type)
        msh_file_path = gmsh_mesh(data_dir, degree)
        cell_type = get_meshio_cell_type(ele_type)
        meshio_mesh = meshio.read(msh_file_path)
        mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict[cell_type])
        return mesh
    
unitcell = Unitcell(ele_type=ele_type)
unitcell.corner_mask
