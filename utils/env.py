import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper
import sys
sys.path.append('../../rl-explanation')
from envs import *


# class MiniGridBallFindingTrainWrapped(MiniGridBallFinding):
#
#     def __init__(self, seed=None):
#         self.env = FullyObsWrapper(super(MiniGridBallFindingTrainWrapped, self).__init__(train=True, seed=seed))
#         self.env = FullyObsWrapper(self.env)
#
# register(
#     id='MiniGrid-BallFindingTrainWrapped-v0',
#     entry_point='utils.env:MiniGridBallFindingTrainWrapped'
# )


def make_env(env_key, seed=None):
    env = FullyObsWrapper(gym.make(env_key))
    env.seed(seed)
    return env
