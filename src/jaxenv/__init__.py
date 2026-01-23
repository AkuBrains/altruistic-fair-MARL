ENV_REGISTRY = {}

ENV_CONFIG_REGISTRY = {}

def register_env(name):
    def decorator(cls):
        ENV_REGISTRY[name] = cls
        return cls
    return decorator


def register_params(env_name):
    def decorator(cls):
        ENV_CONFIG_REGISTRY[env_name] = cls
        return cls
    return decorator

from . import environments, wrappers

def make(env_id: str, **env_settings):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in ENV_REGISTRY.keys():
        raise ValueError(f"{env_id} is not in registered SocialJax environments")
    
    env_kwargs = ENV_CONFIG_REGISTRY[env_id](**env_settings)

    return ENV_REGISTRY[env_id](**env_kwargs)

__all__ = ["make", "ENV_REGISTRY"]
__version__ = "0.0.1"