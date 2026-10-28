
from .base_critic import Critic
from ..utils.environment import Trajectory
from numpy.typing import NDArray

class QCritic(Critic):
    def __init__(self):
        super().__init__()

    def update(self, trajectories : list[Trajectory]):
        pass

    def reward_estimation(self, trajectories : list[Trajectory]):
        pass

class MCQCritic(Critic):
    def __init__(self):
        super().__init__()
    