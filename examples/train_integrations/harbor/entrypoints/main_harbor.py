"""
Main entrypoint for training on Harbor tasks.
"""

import os
import sys
from pathlib import Path

# skyrl-agent is a sibling workspace; make it importable everywhere
_SKYRL_ROOT = Path(__file__).resolve().parents[4]
_AGENT_ROOT = str(_SKYRL_ROOT / "skyrl-agent")
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)
os.environ["SKYRL_AGENT_PATH"] = _AGENT_ROOT

import ray
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.config import SkyRLTrainConfig, GeneratorConfig, get_config_as_yaml_str
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from ..harbor_generator import HarborGenerator
from ..dataset import HarborTaskDataset

# NOTE (sumanthrh): We use a YAML to store the defaults for the Harbor trial configuration
# TODO: Convert to a dataclass
HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class MetaLoopCLIConfig:
    """CLI-passable meta-learning config (mirrors MetaLoopConfig in harbor_generator.py)."""

    enabled: bool = True
    interval_batches: int = 20
    max_candidates: int = 4
    canary_num_tasks: int = 4
    canary_n_samples: int = 3
    override_base: str = ""
    log_dir: str = "/tmp/skyrl-logs"
    llm_model: str = ""
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting and meta-learning."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    meta: MetaLoopCLIConfig = field(default_factory=MetaLoopCLIConfig)


@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    import os as _os, sys as _sys
    _agent = _os.environ.get("SKYRL_AGENT_PATH", "")
    if _agent and _agent not in _sys.path:
        _sys.path.insert(0, _agent)
    exp = HarborExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
