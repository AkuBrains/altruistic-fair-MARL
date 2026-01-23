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
NUM_INNER_STEPS = 100
NUM_OUTER_STEPS = 1

def create_env():
    env = make('common_harvest',
        num_inner_steps=NUM_INNER_STEPS,
        num_outer_steps=NUM_OUTER_STEPS,
        num_agents=NUMS_AGENTS,
    )
    return env
    

        
def get_runner(env):
    # @partial(jax.jit, backend="gpu")
    def runner():
        num_actions = env.action_space(0).n
        rng = jax.random.PRNGKey(123)
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key)


        for o_t in range(NUM_OUTER_STEPS):
            obs, old_state = jax.vmap(env.reset, in_axes=(0))(reset_keys)

            # render each timestep
            for t in range(NUM_INNER_STEPS):
                rng, _rng = jax.random.split(rng)
                actions = jax.random.choice(key=_rng, 
                                            a=num_actions,
                                            shape=(NUM_ENVS, NUMS_AGENTS),
                                            p=jnp.array([1/num_actions]*num_actions))


                obs, state, reward, done, info = jax.vmap(env.step_env, in_axes=(None, 0, 0))(rng, old_state, actions,)
                print('')
                print(f"obs shape : {obs.shape}")

                print('###################')
                jax.debug.print(f'timestep: {t} to {t+1}')
                jax.debug.print('actions:\n {}', actions)
                jax.debug.print('apples remained : {}', (state.grid==3).sum(axis=(1,2 )))
                # print(f'actions: {[action.item() for action in actions]}')
                print("###################")

                old_state = state

    return runner
    
if __name__=="__main__":
    print("######## Env creation test ########")
    env = create_env()
    print(env.action_space(0).n)
    print("Done !\n\n")
    
    print("Python function test :\n")
    runner = get_runner(env)
    runner()
    print("Done !\n\n")
    
    print("######## Jitted function test ######## ")
    runner = get_runner(env)
    jitted_main = jax.jit(runner, backend="cpu")
    jitted_main()
    
    print("Done !")
