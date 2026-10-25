
from .base_critic import Critic
from ..utils.environment import Trajectory
from numpy.typing import NDArray

class QCritic(Critic):
    def __init__(self):
        super().__init__()

    def update(self, trajectories : list[Trajectory]):
        pass

    def state_action_values(self, obs : NDArray):
        pass