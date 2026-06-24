# from gym.envs.registration import register
from gymnasium.envs.registration import register

register(
    id='PreScanEnv-v1',
    entry_point='envs.PS_env:Prescan_env',
)
