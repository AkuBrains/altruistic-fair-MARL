from functools import partial
import os
import sys
import jax.numpy as jnp
import jax
from PIL import Image
from pathlib import Path
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from jaxenv import make

NUM_ENVS = 2
NUMS_AGENTS = 7
GRID_SIZE = (16, 22)
NUM_INNER_STEPS = 5
NUM_OUTER_STEPS = 1

def create_env():
    env = make('clean_up',
        num_inner_steps=NUM_INNER_STEPS,
        num_outer_steps=NUM_OUTER_STEPS,
        num_agents=NUMS_AGENTS,
    )
    print(env.observation_space_shape)
    return env
    

        
def get_runner(env):
    # @partial(jax.jit, backend="gpu")
    def runner():
        rng = jax.random.PRNGKey(123)
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key)

        root_dir = f"random_actions_gif/a{NUMS_AGENTS}_g{GRID_SIZE}_i{NUM_INNER_STEPS}_o{NUM_OUTER_STEPS}"
        path = Path(root_dir + "/state_pics")
        path.mkdir(parents=True, exist_ok=True)


        for o_t in range(NUM_OUTER_STEPS):
            obs, old_state = jax.vmap(env.reset, in_axes=(0))(reset_keys)

            # render each timestep
            for t in range(NUM_INNER_STEPS):
                rng, _rng = jax.random.split(rng)
                actions = jax.random.choice(key=_rng, 
                                            a=env.action_space(0).n,
                                            shape=(NUM_ENVS, NUMS_AGENTS),
                                            p=jnp.array([0.1, 0.1, 0.09, 0.09, 0.09, 0.19, 0.05, 0.1, 0.5]))


                obs, state, reward, done, info = jax.vmap(env.step_env, in_axes=(None, 0, 0))(rng, old_state, actions,)

                print('###################')
                print(f'timestep: {t} to {t+1}')
                print(f'actions:\n{actions}')
                # print(f'actions: {[action.item() for action in actions]}')
                print("###################")

                old_state = state

    return runner
    
if __name__=="__main__":
    print("######## Env creation test ########")
    env = create_env()
    print("Done !\n\n")
    
    print("Python function test :\n")
    runner = get_runner(env)
    runner()
    print("Done !\n\n")
    
    print("######## Jitted function test ######## ")
    runner = get_runner(env)
    jitted_main = jax.jit(runner, backend="gpu")
    jitted_main()
    print("Done !")

