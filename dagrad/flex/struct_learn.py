import torch
from warnings import warn
from ..utils import general_utils
from .modules import loss
import time
import numpy as np


def struct_learn(
    dataset,
    model,
    constrained_solver,
    unconstrained_solver,
    loss_fn,
    dag_fn,
    w_threshold=0.3,
    device="cpu",
    dtype=torch.double,
    verbose: bool = False,
    suppress_warnings: bool = False,
):
    """Learn DAG structure using DAGMA algorithm with linear structural equation."""

    vprint = print if verbose else lambda *a, **k: None
    vwarn = warn if not suppress_warnings else lambda *a, **k: None
    torch.set_default_dtype(dtype)
    device = general_utils.check_device(device)

    n, d = dataset.shape
    if not isinstance(dataset, torch.Tensor):
        dataset = torch.tensor(dataset, dtype=dtype, device=device)

    dataset = dataset.to(dtype=dtype, device=device)
    if isinstance(loss_fn, (loss.MSELoss)):
        dataset = dataset - torch.mean(dataset, dim=0, keepdim=True)

    constrained_solver.dtype = dtype
    constrained_solver.device = device
    constrained_solver.vprint = vprint
    constrained_solver.vwarn = vwarn
    
    unconstrained_solver.dtype = dtype 
    unconstrained_solver.device = device
    unconstrained_solver.vprint = vprint
    unconstrained_solver.vwarn = vwarn 
    
    time_start = time.time()    
    constrained_solver(dataset, model, unconstrained_solver, loss_fn, dag_fn)
    vprint(f"Total Time: {time.time() - time_start}")
    
    W_est = model.adj().detach().cpu().numpy()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est
