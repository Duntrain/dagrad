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
    """
    Perform structural learning of a directed acyclic graph (DAG) from data.
    The function estimates the adjacency matrix of the DAG by solving a constrained optimization problem.
    
    Parameters
    ----------
    dataset : array-like or torch.Tensor
        The input dataset of shape (n_samples, n_features).
    model : object
        The SEM object that contains the adjacency matrix to be learned.
    constrained_solver : object
        Solver for the constrained optimization problem.
    unconstrained_solver : object
        Solver for the unconstrained optimization problem.
    loss_fn : object
        Loss function object defining the fitting criterion.
    dag_fn : object
        Function object implementing the continuous DAG constraint.
    w_threshold : float, optional (default=0.3)
        Threshold value for pruning small weights in the estimated adjacency matrix.
    device : str, optional (default="cpu")
        Device to perform computations on ('cpu' or 'cuda').
    dtype : torch.dtype, optional (default=torch.double)
        Data type for torch tensors.
    verbose : bool, optional (default=False)
        If True, print detailed progress information.
    suppress_warnings : bool, optional (default=False)
        If True, suppress warning messages.
    Returns
    -------
    numpy.ndarray
        The estimated adjacency matrix with small weights thresholded to zero.
    Notes
    -----
    - If the loss function is MSELoss, the data is centered before processing.
    - The function automatically converts input data to torch.Tensor if needed.
    - Both solvers are configured with the specified dtype and device.
    """
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
    print(f"Total Time: {time.time() - time_start}")
    
    W_est = model.adj().detach().cpu().numpy()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est
