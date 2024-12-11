from . import functional as F


class Loss:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Loss Function: {self.__class__.__name__}"

    def eval(self): ...


##
## Regularizers
##


class L1Loss(Loss):
    def eval(self, W):
        return F.l1_loss(W)


class L2Loss(Loss):
    def eval(self, W):
        return F.l2_loss(W)


class MCPLoss(Loss):
    def __init__(self, lmd: float = 0.1, gamma: float = 1.0) -> None:
        self.lmd = lmd
        self.gamma = gamma

    def eval(self, W):
        return F.mcp_loss(W, self.lmd, self.gamma)


##
## Losses
##


class MSELoss(Loss):
    def eval(self, output, target):
        return F.mse_loss(output, target)


class NLLLoss(Loss):
    def eval(self, output, target):
        return F.nll_loss(output, target)


class BCELoss(Loss):
    def eval(self, output, target):
        return F.binary_cross_entropy(output, target)