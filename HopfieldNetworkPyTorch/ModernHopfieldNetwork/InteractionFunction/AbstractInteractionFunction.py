from abc import abstractmethod, ABC
import torch

class AbstractInteractionFunction(ABC):
    
    def __init__(self, n: int):
        """
        Set the interaction vertex for the interaction function.

        :param n: The interaction vertex of the activation function
        """
        
        self.n = n
    
    @abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        pass
    