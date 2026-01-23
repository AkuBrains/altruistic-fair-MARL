import jax.numpy as jnp
import colorsys

from enum import IntEnum




ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap
        [0, 0, 0],  # zap_clean
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],  
        [0, -1, 0],  
        [1, 0, 0],  
        [-1, 0, 0],  
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=jnp.int8,
)

char_to_int = {
    'W': 1,
    ' ': 0,  # empty 0
    'A': 3,  # exist apple, not used in this environment
    'P': 4,  # spawn_point
    'Q': 5,  # spawn_point defence, not used in this environment
    'B': 6, # potential_apple
    'S': 7, # river
    'H': 8, # potential_dirt
    'F': 9, # actual_dirt
    '+': 0, # should be "sand", "shadow_e", "shadow_n"
    'f': 0, # should be "sand", "shadow_e", "shadow_n"
    ";": 0,
    ",": 0,
    "^": 0,
    "=": 0,
    ">": 0,
    "<": 0,
    "~": 7,
    "T": 6,
    

}

def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
        '''
        Rotates agent's observation grid k * 90 degrees, depending on agent's
        orientation.

        Args:
            - agent_loc: jax ndarray of agent's x, y, direction
            - grid: jax ndarray of agent's obs grid

        Returns:
            - jnp.ndarray of new rotated grid.

        '''
        grid = jnp.where(
            agent_loc[2] == 1,
            jnp.rot90(grid, k=1, axes=(0, 1)),
            grid,
        )
        grid = jnp.where(
            agent_loc[2] == 2,
            jnp.rot90(grid, k=2, axes=(0, 1)),
            grid,
        )
        grid = jnp.where(
            agent_loc[2] == 3,
            jnp.rot90(grid, k=3, axes=(0, 1)),
            grid,
        )

        return grid
    

def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

def generate_agent_colors(num_agents):
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)