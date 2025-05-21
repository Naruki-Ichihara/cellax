import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
import gmsh
import scipy
from dataclasses import dataclass
from typing import Callable, List
from cellax.fem.generate_mesh import Mesh

@dataclass
class PeriodicPairing:
    """
    Represents a single periodic boundary condition pair between a 'master' and a 'slave' region.

    Attributes:
        location_master: Function to identify points on the master side of the periodic boundary.
        location_slave: Function to identify points on the slave side of the periodic boundary.
        mapping: Function that maps a point from the master to the corresponding slave location.
        vec: Index of the variable/component (e.g., 0 for x, 1 for y) affected by this condition.
    """
    location_master: Callable[[onp.ndarray], bool]
    location_slave: Callable[[onp.ndarray], bool]
    mapping: Callable[[onp.ndarray], onp.ndarray]
    vec: int

def prolongation_matrix(periodic_pairings: List[PeriodicPairing], mesh: Mesh, vec: int) -> scipy.sparse.csr_array:
    """
    Constructs the prolongation matrix `P` for applying periodic boundary conditions (PBCs)
    in a finite element setting. The matrix `P` enforces constraints by reducing the degrees
    of freedom (DoFs) associated with slave nodes and mapping them to master nodes.

    This technique is commonly used in homogenization problems and representative volume elements (RVEs)
    to impose periodicity between opposing boundaries of the mesh.

    Args:
        periodic_pairings (List[PeriodicPairing]): 
            A list of periodic constraint definitions, each specifying the master/slave regions,
            the geometric mapping between them, and the component index to constrain.
        mesh (Mesh): 
            The finite element mesh object containing nodal coordinates and element connectivity.
        vec (int): 
            The number of degrees of freedom per node (e.g., 2 for 2D vector problems).

    Returns:
        scipy.sparse.csr_array: 
            The prolongation matrix `P` of shape (N, M), where `N` is the total number of DoFs,
            and `M` is the reduced number of independent DoFs after applying periodic constraints.
    """
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []

    for bc in periodic_pairings:
        node_inds_A = onp.argwhere(jax.vmap(bc.location_master)(mesh.points)).reshape(-1)
        node_inds_B = onp.argwhere(jax.vmap(bc.location_slave)(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-5
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = onp.linalg.norm(bc.mapping(point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
        vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * bc.vec

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)
        assert len(node_inds_A) == len(node_inds_B_ordered), \
            f"Mismatch in node pairing: {len(node_inds_A)} master nodes vs {len(node_inds_B_ordered)} slave nodes. Check your mapping."

    offset = 0
    inds_A_list = []
    inds_B_list = []
    for i in range(len(p_node_inds_list_A)):
        inds_A_list.append(onp.array(p_node_inds_list_A[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))
        inds_B_list.append(onp.array(p_node_inds_list_B[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))

    inds_A = onp.hstack(inds_A_list)
    inds_B = onp.hstack(inds_B_list)

    num_total_nodes = len(mesh.points)
    num_total_dofs = num_total_nodes * vec
    N = num_total_dofs
    M = num_total_dofs - len(inds_B)

    reduced_inds_map = onp.ones(num_total_dofs, dtype=onp.int32)
    reduced_inds_map[inds_B] = -(inds_A + 1)
    mask = (reduced_inds_map == 1)
    if onp.count_nonzero(mask) != M:
        raise ValueError(
            f"Inconsistent DoF reduction: expected {M} remaining DoFs "
            f"but found {onp.count_nonzero(mask)} unassigned entries in reduced_inds_map.\n"
            f"Possible cause: some mesh nodes are involved in multiple periodic pairings."
        )
    reduced_inds_map[mask] = onp.arange(M)

    I = []
    J = []
    V = []
    for i in range(num_total_dofs):
        I.append(i)
        V.append(1.0)
        if reduced_inds_map[i] < 0:
            J.append(reduced_inds_map[-reduced_inds_map[i] - 1])
        else:
            J.append(reduced_inds_map[i])

    P_mat = scipy.sparse.csr_array((onp.array(V), (onp.array(I), onp.array(J))), shape=(N, M))

    return P_mat