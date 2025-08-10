# In src/callback.py

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FinalLoggerCallback(BaseCallback):
    """
    A custom callback to manually log the episode statistics prepared by VecMonitor.
    This is a robust way to ensure custom metrics are logged to TensorBoard.
    """

    def __init__(self, verbose=0):
        super(FinalLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'episode' in info:
            episode_stats = info['episode']
            for key, value in episode_stats.items():
                if key not in ['r', 'l', 't']:
                    self.logger.record(f"rollout/{key}", value)
        return True