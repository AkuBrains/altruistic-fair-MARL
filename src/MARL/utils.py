import chex

@chex.dataclass(frozen=True)
class BaseTrainerParams:
    num_envs: int = 20
    total_timesteps: int = 1e6
    trainer_seed: int = 0
    backend: str = "cpu"  # or "gpu"
    num_log_episodes_after_training: int = 10
    debug: bool = False # Print rollout rewards during training
    skip_training: bool = False 
    """skip training and only run "num_log_episodes_after_training" eval episodes."""

