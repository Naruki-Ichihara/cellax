import cellax as cell
import os
import jax
import jax.numpy as np
import gmsh
import meshio
from cellax.fem.problem import Problem
from cellax.fem.generate_mesh import Mesh, box_mesh, rectangle_mesh
from cellax.pbc import PeriodicPairing, prolongation_matrix

N = 3
L = 1.0
class Unitcell_2D(cell.Unitcell):
    def construct_mesh(self):
        mesh = rectangle_mesh(N, N, L, L)
        return mesh
    
unitcell = Unitcell_2D()
unitcell.corner_mask
unitcell.edge_mask
unitcell.face_mask

left = unitcell.edge_function([0], [0], excluding_corner=True)
right = unitcell.edge_function([0], [1], excluding_corner=True)
map = unitcell.mapping(left, right)

print(np.sum(jax.vmap(right)(unitcell.points)))

pair = PeriodicPairing(left, right, map, 0)
P = prolongation_matrix([pair], unitcell.mesh, 1, 0)
print(P.shape)