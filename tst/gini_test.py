import jax
import jax.numpy as jnp


@jax.jit
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

if __name__=="__main__":
    x = jnp.array([1, 1, 10])
    print(gini(x))