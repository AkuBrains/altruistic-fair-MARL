import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import chex
import os
import distrax
import sys
from typing import List, Tuple, NamedTuple
from functools import partial
from jax_tqdm import scan_tqdm

# Assumed imports from your codebase structure

sys.path.append("/home/franck/workspace/MARL/src")
from jaxenv.environments.prisoner_dilemma.ipd import PrisonerDilemma
from jaxenv.wrappers import LogWrapper
from algorithms import trainer, trainer_params
from algorithms.utils import BaseTrainerParams
from utils import logwrapper_callback, F_MAPPO, IPD

HIDDEN_SIZE = 8

class MLP(eqx.Module):
    """MLP encoder for vector observations."""
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    a:str = eqx.field(init=False)
    
    def __init__(self, in_size: int, key: jax.random.PRNGKey, hidden_size: int = HIDDEN_SIZE,  a="relu"):
        k1, k2 = jax.random.split(key)
        self.dense1 = eqx.nn.Linear(in_size, hidden_size, key=k1)
        self.dense2 = eqx.nn.Linear(hidden_size, hidden_size, key=k2)
        self.a=a
        

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = jax.nn.relu if self.a=="relu" else jax.nn.tanh
        x = jnp.ravel(x).astype(jnp.float32)
        x = activation(self.dense1(x))
        x = activation(self.dense2(x))
        return x

class Actor(eqx.Module):
    embedding_net: MLP
    actor_mean: eqx.nn.Linear

    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, key: jax.random.PRNGKey):
        key_mlp, key_mean = jax.random.split(key)
        flat_size = np.prod(obs_shape)
        self.embedding_net = MLP(flat_size, key=key_mlp, a="tanh")
        self.actor_mean = eqx.nn.Linear(HIDDEN_SIZE, action_dim, key=key_mean)

    def __call__(self, obs: jax.Array) -> distrax.Categorical:
        x = self.embedding_net(obs)
        logits = self.actor_mean(x)
        return distrax.Categorical(logits=logits)

class Critic(eqx.Module):
    embedding_net: MLP
    value: eqx.nn.Linear

    def __init__(self, obs_shape: Tuple[int, ...], key: jax.random.PRNGKey):
        key_mlp, key_val = jax.random.split(key)
        flat_size = np.prod(obs_shape)
        self.embedding_net = MLP(flat_size, key=key_mlp)
        self.value = eqx.nn.Linear(HIDDEN_SIZE, 1, key=key_val)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding_net(x)
        value_output = self.value(x)
        return jnp.squeeze(value_output, axis=-1)

# ==========================================
# 3. TRAINER CONFIG
# ==========================================

@chex.dataclass(frozen=True)
@trainer_params(env_name=IPD, algo_name=F_MAPPO)
class MAPPOTrainerParams(BaseTrainerParams):
    learning_rate: float = 2.5e-5
    anneal_lr: bool = True
    gamma: float = 0.99999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    clip_coef_vf: float = 0.5
    ent_coef_start: float = 0.1
    ent_coef_end: float = 0.01
    vf_coef: float = 0.5
    
    num_steps: int = 100
    num_minibatches: int = 4
    update_epochs: int = 4
    a2c_mode: bool = False

    # Runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    
    alpha: float = 0
    fair: bool = True
    backend: str = "gpu"

    def __post_init__(self):
        object.__setattr__(
            self,
            "num_iterations",
            int(self.total_timesteps // self.num_steps // self.num_envs),
        )
        object.__setattr__(
            self,
            "minibatch_size",
            int(self.num_envs * self.num_steps // self.num_minibatches),
        )
        object.__setattr__(
            self, "batch_size", int(self.minibatch_size * self.num_minibatches)
        )

@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    world_state: chex.Array
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array

class TrainState(NamedTuple):
    actors: dict[str, eqx.Module]
    critics: dict[str, eqx.Module]
    optimizer_state: optax.OptState


def rearrange_batch(last_obs, num_envs):
    """
    Adapts IPD observations for MAPPO.
    last_obs shape: (Num_Envs, Num_Agents, H, 2)
    """
    # 1. Observations: Just keep them as is, but maybe flattened?
    # The network expects (H, 2), vmap will handle the batch dims.
    # We essentially just need to ensure the shape lines up for the TrajectoryBatch
    obs_batch = last_obs # (Num_Envs, Num_Agents, H, 2)
    
    # 2. World State: For MAPPO, this is usually the concatenation of all agent obs.
    # Flatten last dimensions: (Num_Envs, Num_Agents, H*2)
    flat_obs = last_obs.reshape(num_envs, last_obs.shape[1], -1)
    
    # Concatenate all agents' obs to form world state: (Num_Envs, Num_Agents*H*2)
    # We want to give the critic the global view.
    # Reshape to (Num_Envs, 1, Num_Agents*H*2)
    global_state = flat_obs.reshape(num_envs, -1)
    global_state = jnp.expand_dims(global_state, 1)
    
    # Broadcast to all agents: (Num_Envs, Num_Agents, Global_State_Dim)
    world_state = jnp.tile(global_state, (1, last_obs.shape[1], 1))
    
    return obs_batch, world_state


@trainer(algo_name=F_MAPPO, env_name=IPD)
def build_trainer(
    env: PrisonerDilemma,  # Expecting the IPD class
    trainer_params: MAPPOTrainerParams = MAPPOTrainerParams(),
    load_model: str = None
):
    config = trainer_params
    eval_env = eqx.tree_at(lambda x: x.train_env, env, False) # In this simple case, same env
    env = LogWrapper(env) # Assumed wrapper

    num_agents = env.num_agents
    obs_shape = env.observation_space_shape # (H, 2)
    
    # Critic input shape: (Num_Agents * H * 2)
    global_shape = (np.prod(obs_shape) * num_agents,)

    # RNG
    rng = jax.random.PRNGKey(config.trainer_seed)
    rng, actor_key, critic_key, reset_key = jax.random.split(rng, 4)

    # Networks
    actor_keys = jax.random.split(actor_key, env.num_agents)
    critic_keys = jax.random.split(critic_key, env.num_agents)
    
    actors = [Actor(obs_shape, env.action_space_shape, key=key) for key in actor_keys]
    critics = [Critic(global_shape, key=key) for key in critic_keys]

    number_of_update_steps = (
        config.num_iterations * config.num_minibatches * config.update_epochs
    )
    learning_rate_schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=1e-6,
        transition_steps=number_of_update_steps,
    )
    ent_coef_schedule = optax.linear_schedule(
        init_value=config.ent_coef_start,
        end_value=config.ent_coef_end,
        transition_steps=number_of_update_steps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=(
                learning_rate_schedule if config.anneal_lr else config.learning_rate
            ),
            eps=1e-5,
        ),
    )

    actors = dict((f"actor{k}", actors[k]) for k in range(len(actors)))
    critics = dict((f"critic{k}", critics[k]) for k in range(len(critics)))

    def apply(models, states):
        def lambda_factory(callable_o):
            def function_o(x):
                return callable_o(x)
            return function_o
        
        indexes = jnp.arange(len(models))
        actors_c = [lambda_factory(model) for model in models.values()]
        # Switch allows applying different network weights per agent index if they are separate
        return jax.vmap(lambda i, s : jax.lax.switch(i, actors_c, s), in_axes=(0, 0))(indexes, states)
    
    optimizer_state = optimizer.init({'actors' : actors, 'critics' : critics})

    train_state = TrainState(
        actors=actors,
        critics=critics,
        optimizer_state=optimizer_state,
    )

    if load_model:
        if not os.path.exists(load_model):
            raise FileNotFoundError(f"Model file not found: {load_model}")
        train_state = eqx.tree_deserialise_leaves(load_model, train_state)
        
    rewards_factor = config["alpha"]*jnp.ones((num_agents, num_agents))+(1-config["alpha"])*jnp.eye(num_agents)

    reset_key = jax.random.split(reset_key, config.num_envs)
    obs_v, env_state_v = jax.vmap(env.reset, in_axes=(0))(reset_key)

    @partial(jax.jit, backend=trainer_params.backend)
    def eval_func(key: chex.PRNGKey, train_state: TrainState):
        def step_env(carry, _):
            rng, obs_v, env_state, done, episode_reward = carry
            
            # obs_v: (Num_Agents, H, 2)
            # In eval we typically run 1 env, so we might need to add batch dim 
            # or handle shapes carefully. Assuming eval runs on single env instance here:
            obs_batch = jnp.expand_dims(obs_v, 0) # (1, Agents, H, 2)
            
            rng, step_key, sample_key = jax.random.split(rng, 3)
            
            # Apply expects (Agents, ...) so we map over the single batch dim
            # But apply() input is (Agents, Obs). 
            # We pass obs_v directly if it's (Agents, H, 2)
            action_dist = apply(train_state.actors, obs_v)
            actions = action_dist.sample(seed=sample_key)
            
            obs_v, env_state, reward, done, info = eval_env.step(
                step_key, env_state, actions
            )
            episode_reward += reward

            return (rng, obs_v, env_state, done, episode_reward), info

        rng, reset_key = jax.random.split(key)
        obs, env_state = eval_env.reset(reset_key)
        done = False
        episode_reward = jnp.zeros(num_agents)

        carry, episode_stats = jax.lax.scan(
            step_env,
            (rng, obs, env_state, done, episode_reward),
            None,
            eval_env.episode_length,
        )
        # Reshape stats if needed
        return carry[-1], episode_stats

    @partial(jax.jit, backend=trainer_params.backend)
    def train_func(rng: chex.PRNGKey = rng):
        
        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)
            
            # Adapted rearrange for IPD
            obs_batch, world_state = rearrange_batch(last_obs, config.num_envs)
            
            # obs_batch: (Num_Envs, Num_Agents, H, 2)
            # apply expects (Num_Agents, H, 2). We vmap over Num_Envs.
            action_dist = jax.vmap(apply, in_axes=(None, 0))(train_state.actors, obs_batch)
            value = jax.vmap(apply, in_axes=(None, 0))(train_state.critics, world_state)
            
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)
            
            
            step_keys = jax.random.split(step_key, config.num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, action)
            
        
            # done is (Num_Envs,) for IPD usually, need broadcast to (Agents, Envs) then Transpose?
            # If done is (Num_Envs,), broadcast to (Num_Envs, Num_Agents)
            broadcasted_done = jnp.broadcast_to(done[:, None], (config.num_envs, num_agents))

            transition = Transition(
                observation=obs_batch,
                world_state=world_state,
                state=env_state,
                action=action,
                reward=reward,
                done=broadcasted_done,
                value=value,
                log_prob=log_prob,
                info=info,
            )
            last_obs = obsv
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        def _calculate_gae(gae_and_next_values, transition):
            gae, next_value = gae_and_next_values
            value, reward, done = transition.value, transition.reward, transition.done
            delta = reward + config.gamma * next_value * (1 - done) - value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, value), (gae, gae + value)

        def _update_epoch(update_state, _):
            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_los_fn(params, trajectory_minibatch, advantages, returns, value_0, count):
                observations = trajectory_minibatch.observation
                world_state = trajectory_minibatch.world_state
                actions = trajectory_minibatch.action
                init_log_prob = trajectory_minibatch.log_prob
                init_value = trajectory_minibatch.value
                
                action_dist = jax.vmap(apply, in_axes=(None, 0))(params["actors"], observations)
                value = jax.vmap(apply, in_axes=(None, 0))(params["critics"], world_state)
                log_prob = action_dist.log_prob(actions)
                entropy = action_dist.entropy().mean(axis=(0))
                
                _advantages = (advantages / (value_0 + 0.5))  @ rewards_factor if config.fair else advantages @ rewards_factor

                if config.a2c_mode:
                    actor_loss = -jnp.mean(log_prob * advantages)
                    value_loss = jnp.mean(jnp.square(value - returns))
                else:
                    ratio = jnp.exp(log_prob - init_log_prob)
                    _advantages = (_advantages - _advantages.mean(axis=0)[None, :]) / (
                        _advantages.std(axis=0)[None, :] + 1e-8
                    )
                    actor_loss1 = _advantages * ratio
                    actor_loss2 = (
                        jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                        * _advantages
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean(axis=0)

                    value_pred_clipped = init_value + (
                        jnp.clip(
                            value - init_value, -config.clip_coef_vf, config.clip_coef_vf
                        )
                    )
                    
                    value_losses = jnp.square(value - returns)
                    value_losses_clipped = jnp.square(value_pred_clipped - returns)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean(axis=0)

                ent_coef = ent_coef_schedule(count)
                total_loss = (
                    actor_loss + config.vf_coef * value_loss - ent_coef * entropy
                ).sum(axis=0)
                return total_loss, (actor_loss, value_loss, ent_coef * entropy)

            def __update_over_minibatch(train_state: TrainState, minibatch):
                trajectory_mb, advantages_mb, returns_mb, value_0_mb = minibatch
                (total_loss, (actor_loss, value_loss, entropy)), grads = __ppo_los_fn(
                    {"actors": train_state.actors, "critics": train_state.critics},
                    trajectory_mb,
                    advantages_mb,
                    returns_mb,
                    value_0_mb,
                    train_state.optimizer_state[1][1].count
                )
                updates, optimizer_state = optimizer.update(
                    grads, train_state.optimizer_state
                )
                new_networks = optax.apply_updates(
                    {"actors": train_state.actors, "critics": train_state.critics}, updates
                )
                train_state = TrainState(
                    actors=new_networks["actors"],
                    critics=new_networks["critics"],
                    optimizer_state=optimizer_state,
                )
                return train_state, (total_loss, actor_loss, value_loss, entropy)

            train_state, trajectory_batch, advantages, returns, rng = update_state
            rng, key = jax.random.split(rng)
            
            value_0 = trajectory_batch.value[0]
            value_0 = jnp.expand_dims(value_0, axis=0)
            value_0 = jnp.tile(value_0, reps=(trajectory_batch.value.shape[0], 1, 1))
            batch = (trajectory_batch, advantages, returns, value_0)
            
            # Flatten batch (Time * Envs * Agents) -> (Samples)
            # Standard MAPPO: Flatten Time and Envs, keep Agents dimension?
            # The harvest implementation flattens everything into a single batch dimension
            # then reshapes. We must be careful with the Agents dimension.
            # Usually: (Time, Envs, Agents, ...)
            # reshape((-1,) + x.shape[2:]) -> (Time*Envs, Agents, ...)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            batch_idx = jax.random.permutation(key, batch[-1].shape[0])
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]),
                shuffled_batch,
            )
            train_state, losses = jax.lax.scan(
                __update_over_minibatch, train_state, minibatches
            )
            return (train_state, trajectory_batch, advantages, returns, rng), losses

        def train_step(runner_state, _):
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )
            rewards = (trajectory_batch.reward.sum(axis=0)).mean(axis=0)
            
            train_state, env_state, last_obs, rng = runner_state
            _, world_state = rearrange_batch(last_obs, config.num_envs)
            
            # jax.debug.print("World state : {}", world_state.shape)
            
            last_value = jax.vmap(apply, in_axes=(None, 0))(train_state.critics, world_state)
            
            _, (advantages, returns) = jax.lax.scan(
                    _calculate_gae,
                    (jnp.zeros_like(last_value), last_value),
                    trajectory_batch,
                    reverse=True,
                    unroll=16,
                )

            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            info = trajectory_batch.info
            m_c = info["M_C"].mean()
            m_d = info["M_D"].mean()
            c_d = info["C_D"].mean()
            
            train_state = update_state[0]
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            metric["rewards"] = rewards
            metric["additional"] = {"m_c" : m_c*100, "m_d" : m_d*100, "c_d": c_d*100}
            
            rng = update_state[-1]
            jax.debug.print("mutual_coop : {}", m_c*100)
            jax.debug.print("mutual_defect : {}", m_d*100)
            jax.debug.print("mixed : {}", c_d*100)
            jax.debug.callback(logwrapper_callback, metric, config.num_envs, config.debug)

            if not config.debug:
                metric = None

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, key = jax.random.split(rng)
        if not trainer_params.skip_training:
            runner_state = (train_state, env_state_v, obs_v, key)
            if not config.debug:
                train_step = scan_tqdm(config.num_iterations)(train_step)
            runner_state, metrics = jax.lax.scan(
                train_step, runner_state, np.arange(config.num_iterations)
            )
            trained_train_state = runner_state[0]
            rng = runner_state[-1]
        else:
            trained_train_state = train_state
            metrics = None

        return {
            "train_state": trained_train_state,
            "train_metrics": metrics
        }

    return train_func, eval_func


if __name__=="__main__":
    seed = jax.random.PRNGKey(42)
    env = PrisonerDilemma()
    
    n = 1
    m_c = []
    m_d = []
    c_d = []
    
    for alpha in jnp.linspace(0, 1, n):
        alpha=1
        print("alpha : ", alpha)
        params = MAPPOTrainerParams(num_envs=100,
                                total_timesteps=1e8,
                                alpha=alpha,
                                fair=True,
                                ent_coef_start=0.01,
                                clip_coef_vf=4,
                                vf_coef=1,
                                gamma=0.999, 
                                learning_rate=1e-4,
                                update_epochs=2,
                                debug=True)
        train_func, eval_func = build_trainer(env, params)
        trainer = jax.jit(train_func, backend="gpu").lower(seed).compile()
        out = trainer(seed)
        train_state = out["train_state"]
        metrics = out["train_metrics"]["additional"]
        m_c.append(metrics["m_c"][-1])
        m_d.append(metrics["m_d"][-1])
        c_d.append(metrics["c_d"][-1])
        
    import matplotlib.pyplot as plt
    
    plt.plot(jnp.linspace(0, 1, n), m_c, label="Mutual Coop.")
    plt.plot(jnp.linspace(0, 1, n), m_d, label="Mutual Def.")
    plt.plot(jnp.linspace(0, 1, n), c_d, label="Mixed")
    plt.xlabel("Altruisme level", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.title("Iterated Prisoner Dilemma (T=100)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("ipd.png", dpi=300, bbox_inches='tight')
    plt.show()
        
    