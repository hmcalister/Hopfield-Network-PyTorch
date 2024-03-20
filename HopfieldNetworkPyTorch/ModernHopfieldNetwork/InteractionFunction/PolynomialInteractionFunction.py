from . import AbstractInteractionFunction
import torch

class PolynomialInteractionFunction(AbstractInteractionFunction):

    def __init__(self, n: int):
        super().__init__(n)    

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X**self.n