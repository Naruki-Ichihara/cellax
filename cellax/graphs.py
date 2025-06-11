import jax.numpy as jnp
from functools import partial
import jax

# ---------------------
# Core distance checker
# ---------------------
def _segment_distance(x, p0, p1):
    v = p1 - p0
    w = x - p0
    t = jnp.clip(jnp.dot(w, v) / jnp.dot(v, v), 0.0, 1.0)
    proj = p0 + t * v
    return jnp.linalg.norm(x - proj)

def universal_graph(x, nodes, edges, radius):
    def check_edge(edge):
        i, j = edge
        return _segment_distance(x, nodes[i], nodes[j]) <= radius
    return jnp.any(jax.vmap(check_edge)(edges)).astype(jnp.uint8)

# ---------------------
# Unit cell definitions
# ---------------------
def fcc_unitcell(scale=1.0):
    base = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    offsets = jnp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    nodes = jnp.concatenate([base + o for o in offsets], axis=0) * scale
    edges = jnp.array([[0, 1], [0, 2], [0, 3], [1, 6], [2, 5], [3, 4]])
    return nodes, edges

def bcc_unitcell(scale=1.0):
    nodes = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ]) * scale
    edges = jnp.array([[i, 8] for i in range(8)])
    return nodes, edges

def simple_cube_unitcell(scale=1.0):
    nodes = jnp.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1],
    ]) * scale
    edges = jnp.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ])
    return nodes, edges


# ---------------------
# Functional interfaces
# ---------------------
def fcc(radius=0.1, scale=1.0):
    nodes, edges = fcc_unitcell(scale)
    return partial(universal_graph, nodes=nodes, edges=edges, radius=radius)

def bcc(radius=0.1, scale=1.0):
    nodes, edges = bcc_unitcell(scale)
    return partial(universal_graph, nodes=nodes, edges=edges, radius=radius)

def sc(radius=0.1, scale=1.0):
    nodes, edges = simple_cube_unitcell(scale)
    return partial(universal_graph, nodes=nodes, edges=edges, radius=radius)