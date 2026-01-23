import jax
import jax.numpy as jnp


def _calculate_gae(gae_and_next_values, transition):
            gae, next_value = gae_and_next_values
            value, reward, gamma, done = (
                transition.value,
                transition.reward, # if truncated, this is already raised
                transition.discount,
                transition.done,
            )
            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * 0.95 * (1 - done) * gae
            return (gae, value), (gae, gae + value)

def test_gae(trajectory_batch, last_value):
    _, (advantages, returns) = jax.lax.scan(
                    _calculate_gae,
                    (jnp.zeros_like(last_value), last_value),
                    trajectory_batch,
                    reverse=True,
                    unroll=16,
                )
    return advantages, returns