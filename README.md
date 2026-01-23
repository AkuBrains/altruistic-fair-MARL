

# Fair MARL: A Framework for Fairness in Multi-Agent Reinforcement Learning

We introduce a novel framework and algorithms for promoting fairness among agents in cooperative and competitive multi-agent reinforcement learning settings.
The project is built on top of the Jax version of [RICE-N](https://github.com/mila-iqia/climate-cooperation-competition/tree/GAIA_jax) and incorporates environments from [SocialJax](https://github.com/cooperativex/SocialJax).

### Built With

This project is built using modern deep reinforcement learning libraries.

  * Jax
  * Equinox
  * Cuda12
-----

## Getting Started

Follow these instructions to set up the project on your local machine for development and testing purposes.

### Prerequisites

Need Python 3.11+ and Cuda12 installed for GPU acceleration. We recommend using a virtual environment.

### Installation

  **Install dependencies**
    Install all the required packages using the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```

-----

## Usage

This section provides instructions on how to train and evaluate the models.

### Training a Model

To train a fair MARL algorithm on a specific environment, run the `train_suite.py` script with the desired configuration.

```sh
python src/train_suite.py --config "src/configs/config_file.yaml" 
```

In the config file, there are four parameters : 
```sh
fair: bool # True to use fair objective and False to use the Util. one
alpha: float # Between 0 and 1, it is altruism level 
a2c_mode: bool # True to use MAA2C and False to use the one defined in the config file
algorithm: str # The name of the algo defined in src/util/const.py  
```

Models are saved at the root of the project under the `saved_models` folder. 



### Evaluating a Trained Model

To evaluate a pre-trained model and visualize its performance, use the `src/draw_eval.py`
```sh
python src/draw_eval.py --models "/path/to/eqx/file.eqx" -n_agent 7 --n_episode 1 --step_env 100 --seed 42
```

