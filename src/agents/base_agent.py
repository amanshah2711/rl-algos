
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, target_policy):
        super().__init__()