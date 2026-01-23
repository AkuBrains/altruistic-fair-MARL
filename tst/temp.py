import jax 
import jax.numpy as jnp 

@jax.jit
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

A = jnp.ones((100,10))
A.at[50, 6].set(0)
print(find_first_zero_per_column(A))