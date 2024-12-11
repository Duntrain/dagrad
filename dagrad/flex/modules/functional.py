import torch


##
## Regularization functions
##


def l1_loss(W):
    return torch.sum(torch.abs(W))


def l2_loss(W):
    return 0.5 * torch.sum(W**2)


def mcp_loss(W, lmd: float = 0.1, gamma: float = 1.0):
    cond = torch.abs(W) <= gamma
    reg_val = lmd * torch.where(cond, torch.abs(W) - W**2 / (2 * gamma), gamma / 2)
    return reg_val.sum()


##
## Loss functions
##


def mse_loss(output, target):
    n, _ = target.shape
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def nll_loss(output, target):
    n, d = target.shape
    loss = 0.5 * d * torch.log(1.0 / n * torch.sum((output - target) ** 2))
    return loss


def binary_cross_entropy(output, target):
    n, _ = target.shape
    loss = (
        -1.0
        / n
        * torch.sum(target * torch.log(output) + (1 - target) * torch.log(1 - output))
    )
    return loss


##
## DAG functions
##


def exp(W):
    return torch.trace(torch.linalg.matrix_exp(W * W)) - W.shape[0]


def log_det(W, s: float = 1.0):
    d = W.shape[0]
    device, dtype = W.device, W.dtype
    Id = torch.eye(d, dtype=dtype, device=device)
    s = torch.tensor(s, device=device, dtype=dtype)
    h = -torch.slogdet(s * Id - W * W)[1] + d * torch.log(s)
    return h


def poly(W):
    d, device, dtype = W.shape[0], W.device, W.dtype
    M = torch.eye(d, device=device, dtype=dtype) + W * W / d
    E = torch.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d
    return h


def exp_abs(W):
    return torch.trace(torch.linalg.matrix_exp(torch.abs(W))) - W.shape[0]


def log_det_abs(W, s: float = 1.0):
    d = W.shape[0]
    device, dtype = W.device, W.dtype
    Id = torch.eye(d, dtype=dtype, device=device)
    s = torch.tensor(s, device=device, dtype=dtype)
    h = -torch.slogdet(s * Id - torch.abs(W))[1] + d * torch.log(s)
    return h


def poly_abs(W):
    d, device, dtype = W.shape[0], W.device, W.dtype
    M = torch.eye(d, device=device, dtype=dtype) + torch.abs(W) / d
    E = torch.linalg.matrix_power(M, d - 1)
    return (E.T * M).sum() - d
