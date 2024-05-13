from . import AbstractInteractionFunction
import torch

class LeakyRectifiedPolynomialInteractionFunction(AbstractInteractionFunction):

    def __init__(self, n: int, negativeSlope: float = 1e-2):
        super().__init__(n)    
        self.negativeSlope = negativeSlope

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.where(X<=0, self.negativeSlope*X, X**self.n)