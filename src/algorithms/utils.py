import chex
import jax.numpy as jnp

@chex.dataclass(frozen=True)
class BaseTrainerParams:
    num_envs: int = 20
    total_timesteps: int = 1e6
    trainer_seed: int = 0
    backend: str = "cpu"  # or "gpu"
    num_log_episodes_after_training: int = 10
    debug: bool = False # Print rollout rewards during training
    skip_training: bool = False 
    """skip training and only run "num_log_episodes_after_training" eval episodes."""
    
    
def gini(x: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Gini index of a 1D array.

    Args:
        x: A JAX array of non-negative values.

    Returns:
        The Gini index, a float between 0 and 1.
    """
    # The array must be non-negative
    x = jnp.abs(x)
    
    # Sort the array in ascending order
    x_sorted = jnp.sort(x)
    
    n = x_sorted.shape[0]
    
    # A more numerically stable and efficient formula
    # G = (2 * sum(i * x_i for i=1 to n)) / (n * sum(x_i)) - (n + 1) / n
    i = jnp.arange(1, n + 1)
    
    numerator = 2 * jnp.sum(i * x_sorted)
    denominator = n * jnp.sum(x_sorted)
    
    # Handle the case where the denominator is zero (all values are zero)
    gini_index = jnp.where(denominator == 0, 0., (numerator / denominator) - ((n + 1) / n))
    
    return gini_index

def find_first_zero_per_column(matrix, length=-1):
  """
  Finds the row index of the first zero in each column of a JAX matrix.

  Args:
    matrix: A 2D JAX numpy array.

  Returns:
    A 1D JAX numpy array of shape (matrix.shape[1],), where each
    element is the row index of the first zero found in that column.
    Returns -1 for any column that contains no zeros.
  """
  is_zero_mask = (matrix == 0)

  first_zero_indices = jnp.argmax(is_zero_mask, axis=0)

  has_any_zero = jnp.any(is_zero_mask, axis=0)

  result = jnp.where(has_any_zero, first_zero_indices, length)

  return result


