import jax
import equinox as eqx
from typing import List
import distrax
import jax.numpy as jnp

from chex import dataclass

from jaxenv.environments.rice import OBSERVATIONS, ACTION_MASK, GLOBAL_OBS
from networks import network

BIG_NUMBER_NEG = -1e7


class BaseNetwork(eqx.Module):
    cnn: bool = False
    def __init__(self, key, in_shape: int, out_shape: int | List[int]):
        pass


@network
class ActorNetworkMultiDiscrete(BaseNetwork):
    """'
    Actor network for a multidiscrete output space
    """

    layers: list
    output_heads: list

    def __init__(self, key, in_shape: int, out_shape: List[int], hidden_layers: List[int]):
        super().__init__(key, in_shape, out_shape)
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(
                eqx.nn.Linear(feature, hidden_layers[i + 1], key=keys[i])
            )

        multi_discrete_heads_keys = jax.random.split(keys[-1], len(out_shape))
        self.output_heads = [
            eqx.nn.Linear(hidden_layers[-1], action, key=multi_discrete_heads_keys[i])
            for i, action in enumerate(out_shape)
        ]
        if len(set(out_shape)) == 1:  # all output shapes are the same, vmap
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )
        else:
            raise NotImplementedError(
                "Different output shapes detected. Call function does not account for this yet"
            )

    def __call__(self, x):
        if isinstance(x, dict):
            action_mask = x[ACTION_MASK]
            x = x[OBSERVATIONS]
        else:
            action_mask = None

        def forward(head, x):
            return head(x)

        for layer in self.layers:
            x = jax.nn.tanh(layer(x))
        logits = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)

        if action_mask is not None:  # mask the logits
            logit_mask = jnp.ones_like(logits) * BIG_NUMBER_NEG
            
            logit_mask = logit_mask * (1 - action_mask)
            logits = logits + logit_mask

        return distrax.Categorical(logits=logits)


@network
class Q_CriticNetworkMultiDiscrete(BaseNetwork):
    """'
    Critic network that outputs values for each action
    an array of Q-values
    """

    layers: list
    output_heads: list

    def __init__(self, key, in_shape: int, out_shape: List[int], hidden_layers: List[int]):
        super().__init__(key, in_shape, out_shape)
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(
                eqx.nn.Linear(feature, hidden_layers[i + 1], key=keys[i])
            )

        multi_discrete_heads_keys = jax.random.split(keys[-1], len(out_shape))
        self.output_heads = [
            eqx.nn.Linear(hidden_layers[-1], action, key=multi_discrete_heads_keys[i])
            for i, action in enumerate(out_shape)
        ]
        if len(set(out_shape)) == 1:  # all output shapes are the same, vmap
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )
        else:
            raise NotImplementedError(
                "Different output shapes detected. Call function does not account for this yet"
            )

    def __call__(self, x):
        if isinstance(x, dict):
            x = x[OBSERVATIONS]

        def forward(head, x):
            return head(x)

        for layer in self.layers:
            x = jax.nn.tanh(layer(x))
        output = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)
        return output


@network
class CriticNetwork(BaseNetwork):
    """
    Critic network with a single output
    Used for example to output V when given a state
    or Q when given a state and action
    """

    layers: list
    entry_key: str = eqx.field(static=True) 

    def __init__(self, key, in_shape, hidden_layers: List[int], out_shape=None, entry_key=OBSERVATIONS):
        super().__init__(key, in_shape, out_shape)
        self.entry_key = entry_key
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [  # init with first layer
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(
                eqx.nn.Linear(feature, hidden_layers[i + 1], key=keys[i])
            )
        self.layers.append(eqx.nn.Linear(hidden_layers[-1], 1, key=keys[-1]))

    def __call__(self, x):
        if isinstance(x, dict):
            x = x[self.entry_key]
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)
