import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time
from petsc4py import PETSc

from cellax.fem import logger

from jax import config
config.update("jax_enable_x64", True)

