from . import AbstractInteractionFunction
import torch

class RectifiedPolynomialInteractionFunction(AbstractInteractionFunction):

    def __init__(self, n: int):
        super().__init__(n)    

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.where(X<=0, 0, X**self.n)