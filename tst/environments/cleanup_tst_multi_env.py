import os
import sys
from typing import Tuple

import distrax
import jax.numpy as jnp
import jax
import math
import equinox as eqx 


from pathlib import Path
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from jaxenv import make
from jaxenv.environments.cleanup.clean_up import CleanUp
from algorithms.MAPPO.mappo_cnn_cleanup import CNN, Actor, Critic, TrainState

    
def eval():
    
    num_agents=7
    grid_size = (19,28)
    num_inner_steps=100
    num_outer_steps=1
    rng = jax.random.PRNGKey(42)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    load_model = "/home/franck/workspace/MARL/saved_models/FAIR_MAPPO_clean_up_1759195861.3512757.eqx"
    
    
    num_agents = env.num_agents
    
    # Input shape 
    obs_shape = env.observation_space_shape
    global_shape = (*obs_shape[:-1], obs_shape[-1] * env.num_agents)

    # rng keys
    rng, actor_key, critic_key, reset_key = jax.random.split(rng, 4)
    
    actor_keys = jax.random.split(actor_key, env.num_agents)
    critic_keys = jax.random.split(critic_key, env.num_agents)
    actors = [Actor(obs_shape, env.action_space_shape, key=key) for key in actor_keys]
    critics = [Critic(global_shape, key=key) for key in critic_keys]
    
    actors = dict((f"actor{k}", actors[k]) for k in range(len(actors)))
    critics = dict((f"critic{k}", critics[k]) for k in range(len(critics)))
    
    train_state = TrainState(
        actors=actors,
        critics=critics,
        optimizer_state=None,
    )
    
    train_state = eqx.tree_deserialise_leaves(load_model, train_state)
    
    actors = train_state.actors

    rng, _rng, key = jax.random.split(rng, 3)


    root_dir = f"eval_gif/a{num_agents}_g{grid_size}_i{num_inner_steps}_o{num_outer_steps}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)
    
        
    for o_t in range(num_outer_steps):
        obs, old_state = env.reset(_rng)

        # render each timestep
        pics = []
        pics1 = []
        pics2 = []

        img = env.render(old_state)
        Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
        pics.append(img)

        for t in range(num_inner_steps):
            obs= jnp.transpose(obs, (0, 3, 1, 2))
            rng, sample_key, _rng = jax.random.split(rng, 3)
            
            action_dist = het_models_forward(actors, obs)
            actions, _ = action_dist.sample_and_log_prob(seed=sample_key)


            obs, state, reward, done, info = env.step_env(rng, old_state, actions,)

            print('###################')
            print(f'timestep: {t} to {t+1}')
            print(f'actions: {[action.item() for action in actions]}')
            print(f"Reward: {reward}")
            print(f"Info: {info}")
            print("###################")
            
            img = env.render(state)
            Image.fromarray(img).save(
                f"{root_dir}/state_pics/state_{t+1}.png"
            )
            pics.append(img)

            old_state = state

        # create and save gif
        print("Saving GIF")
        pics = [Image.fromarray(img) for img in pics]
        pics[0].save(
        f"{root_dir}/state_outer_step_{o_t+1}.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
        )
        
        
    
def main():
    
    num_agents=7
    grid_size = (19,28)
    num_inner_steps=100
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    print(env.state_space_shape)
    rng, _rng, key = jax.random.split(rng, 3)


    root_dir = f"random_actions_gif/a{num_agents}_g{grid_size}_i{num_inner_steps}_o{num_outer_steps}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)
    
        
    for o_t in range(num_outer_steps):
        obs, old_state = env.reset(_rng)

        # render each timestep
        pics = []
        pics1 = []
        pics2 = []

        img = env.render(old_state)
        Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
        pics.append(img)

        for t in range(num_inner_steps):
            rng, _rng = jax.random.split(rng)
            actions = jax.random.choice(key=_rng, 
                                        a=env.action_space(0).n,
                                        shape=(num_agents,),
                                        p=jnp.array([0.1, 0.1, 0.09, 0.09, 0.09, 0.19, 0.05, 0.1, 0.5]))


            obs, state, reward, done, info = env.step_env(rng, old_state, actions,)

            print('###################')
            print(f'timestep: {t} to {t+1}')
            print(f'actions: {[action.item() for action in actions]}')
            print(f"Reward: {reward}")
            print(f"Info: {info}")
            print("###################")
            
            img = env.render(state)
            Image.fromarray(img).save(
                f"{root_dir}/state_pics/state_{t+1}.png"
            )
            pics.append(img)

            old_state = state

        # create and save gif
        print("Saving GIF")
        pics = [Image.fromarray(img) for img in pics]
        pics[0].save(
        f"{root_dir}/state_outer_step_{o_t+1}.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
        )
        
        
def het_models_forward(models, states):
        def lambda_factory(callable_o):
            def function_o(x):
                return callable_o(x)
            return function_o
        
        indexes = jnp.arange(len(models))
        actors_c = [lambda_factory(model) for model in models.values()]
        return jax.vmap(lambda i, s : jax.lax.switch(i, actors_c, s), in_axes=(0, 0))(indexes, states)
    

def apply(models, inputs):
    def _apply(model, i):
        return model(i)
    return jax.vmap(_apply, in_axes=(0,0))(models, inputs)
        
def test_actor():
    num_agents=7
    grid_size = (19,28)
    num_inner_steps=10
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    print(env.observation_space_shape)
    rng, _rng, key = jax.random.split(rng, 3)
    actor = Actor(env.observation_space_shape, env.action_space_shape, key=key)
    obs, old_state = env.reset(_rng)
    print(obs.shape)
    obs_temp = jnp.transpose(obs, (0, 3, 1, 2))
    print(obs.shape)
    action = actor(obs_temp[0])
    print(action)
    
    
def test_actor_multienv():
    num_agents=7
    num_env=2
    num_inner_steps=10
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    
    print(env.observation_space_shape)
    rng, _rng, key = jax.random.split(rng, 3)
    envs_rng = jax.random.split(_rng, num_env)
    
    actor = Actor(env.observation_space_shape, env.action_space_shape, key=key)
    
    obs, old_state = jax.vmap(env.reset)(envs_rng)
    print(obs.shape)
    obs_temp = jnp.transpose(obs, (0, 1, 4, 2, 3))
    print(obs.shape)
    def apply(x):
        return actor(x)
    action = jax.vmap(apply)(obs_temp[:, 0])
    print(action.logits.shape)

def test_multiactor_env():
    num_agents=7
    num_env=1
    num_inner_steps=10
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    
    print(env.observation_space_shape)
    rng, _env_rng, _actor_rng = jax.random.split(rng, 3)
    actor_keys = jax.random.split(_actor_rng, num_agents)
    
    actors = [Actor(env.observation_space_shape, env.action_space_shape, key=key) for key in actor_keys]
    actors = dict((f"actor{k}", actors[k]) for k in range(len(actors)))
    
    obs, old_state = env.reset(_env_rng)
    print(obs.shape)
    obs_temp = jnp.transpose(obs, (0, 3, 1, 2))
    print(obs_temp.shape)
 
    actions = het_models_forward(actors, obs_temp)
    print(actions.logits.shape)
    
def test_multiactor_multienv():
    num_agents=7
    num_env=2
    num_inner_steps=10
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    
    print(env.observation_space_shape)
    rng, _env_rng, _actor_rng = jax.random.split(rng, 3)
    envs_rng = jax.random.split(_env_rng, num_env)
    actor_keys = jax.random.split(_actor_rng, num_agents)
    
    actors = [Actor(env.observation_space_shape, env.action_space_shape, key=key) for key in actor_keys]
    actors = dict((f"actor{k}", actors[k]) for k in range(len(actors)))
    
    obs, old_state = jax.vmap(env.reset)(envs_rng)
    print(obs.shape)
    obs_temp = jnp.transpose(obs, (0, 1, 4, 2, 3))
    print(obs_temp.shape)
    
    actions = jax.vmap(het_models_forward, in_axes=(None, 0))(actors, obs_temp)
    print(actions.logits.shape)
    
def test_multicritic_multienv():
    num_agents=7
    num_env=2
    num_inner_steps=10
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    rng, _env_rng, _actor_rng = jax.random.split(rng, 3)
    obs_shape = env.observation_space_shape
    print(obs_shape)
    global_shape = (*obs_shape[:-1], obs_shape[-1] * env.num_agents)
    print(global_shape)
    envs_rng = jax.random.split(_env_rng, num_env)
    actor_keys = jax.random.split(_actor_rng, num_agents)
    
    critics = [Critic(global_shape, key=key) for key in actor_keys]
    critics = dict((f"critic{k}", critics[k]) for k in range(len(critics)))
    
    obs, old_state = jax.vmap(env.reset)(envs_rng)

    world_state = jnp.transpose(obs, (0,2,3,1,4)).reshape(num_env, *(env.observation_space()[0]).shape[:-1], -1)
    world_state = jnp.expand_dims(world_state, axis=0)
    world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
    world_state = jnp.transpose(world_state, (1, 0, 4, 2, 3))
    print(world_state.shape)
    values = apply(critics, world_state[0])
    values = jax.vmap(apply, in_axes=(None, 0))(critics, world_state)
    # values = jax.vmap(het_models_forward, in_axes=(None, 0))(critics, world_state)
    print(values.shape)
    
def test_combine():
    num_agents=7
    num_env=2
    num_inner_steps=10
    num_outer_steps=1
    rng = jax.random.PRNGKey(123)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    rng, _env_rng, _actor_rng = jax.random.split(rng, 3)
    obs_shape = env.observation_space_shape
    print(obs_shape)
    global_shape = (*obs_shape[:-1], obs_shape[-1] * env.num_agents)
    print(global_shape)
    envs_rng = jax.random.split(_env_rng, num_env)
    actor_keys = jax.random.split(_actor_rng, num_agents)
    
    critics = [Critic(global_shape, key=key) for key in actor_keys]
    stacked_critics = eqx.combine(*critics)
    
    obs, old_state = jax.vmap(env.reset)(envs_rng)

    world_state = jnp.transpose(obs, (0,2,3,1,4)).reshape(num_env, *(env.observation_space()[0]).shape[:-1], -1)
    world_state = jnp.expand_dims(world_state, axis=0)
    world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
    world_state = jnp.transpose(world_state, (1, 0, 4, 2, 3))
    print(world_state.shape)
    
    
    values = jax.vmap(stacked_critics)(world_state)
    print(values.shape)        
    
def apply(env: CleanUp, rng, old_state, actions):
    return env.step_env(rng, old_state, actions,)


if __name__=="__main__":
    eval()