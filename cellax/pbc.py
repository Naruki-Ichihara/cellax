import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
import gmsh
import scipy
from dataclasses import dataclass
from typing import Callable, Iterable
from cellax.fem.generate_mesh import Mesh
from cellax.unitcell import Unitcell
import itertools

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

def prolongation_matrix(periodic_pairings: Iterable[PeriodicPairing], mesh: Mesh, vec: int, offset: int=0) -> scipy.sparse.csr_array:
    """
    Constructs the prolongation matrix `P` for applying periodic boundary conditions (PBCs)
    in a finite element setting. The matrix `P` maps reduced (independent) degrees of freedom (DoFs)
    to the full set of DoFs, including those constrained by periodicity.

    This is commonly used in homogenization problems and representative volume elements (RVEs),
    where periodicity is imposed by expressing slave DoFs as linear combinations of master DoFs.

    Specifically, given a reduced DoF vector `ū`, the full DoF vector `u` satisfying periodic constraints
    is reconstructed as `u = P @ ū`. Conversely, residuals and system matrices in full space can be
    projected into reduced space via `P.T`.

    Args:
        periodic_pairings (Iterable[PeriodicPairing]): 
            A list of periodic constraint definitions, each specifying the master/slave regions,
            the geometric mapping between them, and the component index to constrain.
        mesh (Mesh): 
            The finite element mesh object containing nodal coordinates and element connectivity.
        vec (int): 
            The number of degrees of freedom per node (e.g., 2 for 2D vector problems).
        offset (int, optional, default=0):
            The offset to be added to the node indices when constructing the prolongation matrix.

    Returns:
        scipy.sparse.csr_array: 
            The prolongation matrix `P` of shape `(N, M)`, where:
            - `N` is the total number of DoFs before applying periodic constraints
                - `M` is the number of independent DoFs after applying the constraints

            The matrix can be used as:
                - `u = P @ ū` to expand a solution from reduced to full space
                - `R̄ = P.T @ R` to project a residual from full to reduced space
                - `J̄ = P.T @ J @ P` to obtain the reduced system Jacobian
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

def periodic_bc_3D(
    unitcell: Unitcell,
    vec: int = 1,
    dim: int = 3):

    L = unitcell.ub - unitcell.lb

    face_pairs = []
    for axis in range(dim):
        master_fn = unitcell.face_function(axis, 0, excluding_edge=True)
        slave_fn = unitcell.face_function(axis, L[axis], excluding_edge=True)
        for i in range(vec):
            face_pairs.append(PeriodicPairing(
                master_fn,
                slave_fn,
                unitcell.mapping(master_fn, slave_fn), [i]))

    edge_pairs = []
    for axes in [[1, 2], [0, 2], [0, 1]]:
        for values in [[L[axes[0]], 0], [L[axes[0]], L[axes[1]]], [0, L[axes[1]]]]:
            master_fn = unitcell.edge_function(axes, [0, 0], excluding_corner=True)
            slave_fn = unitcell.edge_function(axes, values, excluding_corner=True)
            for i in range(vec):
                edge_pairs.append(
                    PeriodicPairing(
                        master_fn,
                        slave_fn,
                        unitcell.mapping(master_fn, slave_fn), [i]))
                
    corner_origin = unitcell.lb
    corner_pairs = []
    for corner in itertools.product(*[[corner_origin[i], corner_origin[i] + L[i]] for i in range(dim)]):
        if np.allclose(np.array(corner), corner_origin):
            continue
        master_fn = unitcell.corner_function(corner_origin)
        slave_fn = unitcell.corner_function(corner)
        for i in range(vec):
            corner_pairs.append(
                PeriodicPairing(
                    master_fn,
                    slave_fn,
                    unitcell.mapping(master_fn, slave_fn),
                    [i]
                )
            )
    return corner_pairs + edge_pairs + face_pairs