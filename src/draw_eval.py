import os
import sys
import argparse


import jax.numpy as jnp
import jax
import equinox as eqx 


from pathlib import Path
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from jaxenv import make
from jaxenv.environments.cleanup.clean_up import CleanUp
from algorithms.MAPPO.mappo_cnn_cleanup import CNN, Actor, Critic, TrainState


def het_models_forward(models, states):
        def lambda_factory(callable_o):
            def function_o(x):
                return callable_o(x)
            return function_o
        
        indexes = jnp.arange(len(models))
        actors_c = [lambda_factory(model) for model in models.values()]
        return jax.vmap(lambda i, s : jax.lax.switch(i, actors_c, s), in_axes=(0, 0))(indexes, states)

    
def eval(model_path, num_agents=7, num_inner_steps=100, num_outer_steps=1, seed=42):
    grid_size = (19,28)
    rng = jax.random.PRNGKey(seed)
    env: CleanUp = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
    
    load_model = model_path
    
    
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
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Test and visualize how the trained models evolve in CleanUp."
    )
    parser.add_argument('-c', '--models', type=str, help="Path to the saved models, the file should end with '.eqx'")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generator")
    parser.add_argument('--step_env', type=int, default=100,  help="Num steps for a single episode")
    parser.add_argument('--n_episode', type=int, default=1,  help="Num episodes")
    parser.add_argument('-a', '--n_agent', type=int, default=7,  help="Number of agents, should be consistent with the trained models")

    args = parser.parse_args()

    eval(args.models, num_agents=args.n_agent, num_inner_steps=args.step_env, num_outer_steps=args.n_episode, seed=args.seed)
