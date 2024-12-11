import copy
from typing import List, Optional, Union
import torch
# import tqdm as tqdm
from tqdm.auto import tqdm

class ConstrainedSolver:
    def __init__(self):
        self.device = None
        self.dtype = None
        self.verbose = None
        self.suppress_warnings = None

    def __call__(self, *args, **kwds):
        return self.solve(*args, **kwds)

    def __repr__(self):
        return f"Constrained Solver: {self.__class__.__name__}"

    def solve(self):
        raise NotImplementedError


class PathFollowing(ConstrainedSolver):
    def __init__(
        self,
        num_iter: int,
        mu_init: float,
        mu_scale: float,
        logdet_coeff: Optional[float] = None,
        num_steps: Optional[Union[List[int], int]] = None,
        weight_decay: Union[List[float], float] = 0,
        l1_coeff: Union[List[float], float] = 0,
    ):
        super().__init__()

        self.num_iter = num_iter
        self.num_steps = num_steps
        self.mu = mu_init
        self.mu_scale = mu_scale
        self.s = logdet_coeff
        self.weight_decay = weight_decay
        self.l1_coeff = l1_coeff

    def validate_inputs(self):
        if isinstance(self.num_steps, list):
            if len(self.num_steps) < self.num_iter:
                missing_steps = self.num_iter - len(self.num_steps)
                self.vwarn(
                    f"Using the first value from num_steps for the first {missing_steps} iterations"
                )
                self.num_steps = missing_steps * [self.num_steps[0]] + self.num_steps
        elif isinstance(self.num_steps, (int, float)):
            self.num_steps = self.num_iter * [self.num_steps]
        else:
            self.vwarn(
                "num_steps is None, will use the default value from Unconstrained Solver"
            )
            if self.unconstrained_solver.num_steps is None:
                raise ValueError("num_steps is None in Unconstrained Solver")
            else:
                self.num_steps = self.num_iter * [self.unconstrained_solver.num_steps]

        if isinstance(self.s, list):
            missing_s = self.num_iter - len(self.s)
            if len(self.s) < self.num_iter:
                self.vwarn(
                    f"Using the last value from s for the last {missing_s} iterations"
                )
                self.s = self.s + missing_s * [self.s[-1]]
        elif isinstance(self.s, (int, float)):
            self.s = self.num_iter * [self.s]
        else:
            ValueError("s should be a list, int, or float.")
        
        if self.l1_coeff > 0:
            # make sure the function l1_loss is implemented in the model
            if not hasattr(self.model, "l1_loss"):
                raise ValueError("Model does not have l1_loss method")

    def solve(self, dataset, model, unconstrained_solver, loss_fn, dag_fn):
        torch.set_default_dtype(self.dtype)
        self.model = model
        self.unconstrained_solver = unconstrained_solver
        self.vprint("Using Path Following Scheme to Solve the DAG constrained Problem")
        self.validate_inputs()

        for i in tqdm(range(self.num_iter)):
            dag_fn.s = self.s[i]  # used only for logdet function

            def new_loss(output, target):
                total_loss = loss_fn(output, target)
                if self.l1_coeff > 0:
                    total_loss += self.l1_coeff * model.l1_loss()
                if self.weight_decay > 0:
                    l2_loss = torch.tensor(0.0).to(self.device)
                    for param in model.parameters():
                        l2_loss += torch.sum(param**2)
                    total_loss += 0.5 * self.weight_decay * l2_loss
                return self.mu * total_loss + dag_fn(model.adj())

            success = False
            model_copy = copy.deepcopy(model)
            lr_scale, lr_scheduler = None, False

            while success is False:
                self.vprint(
                    f"\nSolving unconstrained problem #{i} with mu={self.mu}, s={dag_fn.s} for {self.num_steps[i]} steps"
                )
                unconstrained_solver.num_steps = self.num_steps[i]
                success = unconstrained_solver(
                    dataset, model, new_loss, dag_fn, lr_scale, lr_scheduler
                )

                if success is False:
                    model.load_state_dict(
                        model_copy.state_dict().copy()
                    )  # .to(device)?
                    lr_scale = 0.5 if lr_scale is None else 0.5 * lr_scale
                    lr_scheduler = True
                    if lr_scale < 1e-6:
                        break  # lr is too small
                    dag_fn.s = 1  # only useful when dag_fn is the logdet function
            self.mu *= self.mu_scale


class AugmentedLagrangian(ConstrainedSolver):
    def __init__(
        self,
        num_iter: int = 20,
        num_steps: Optional[Union[List[int], int]] = None,
        alpha_init: float = 0.0,
        rho_init: float = 1.0,
        rho_scale: float = 10.0,
        rho_max: float = 1e8,
        weight_decay: float = 0.0,
        l1_coeff: float = 0.0,
        h_tol: float = 1e-8
    ):
        super().__init__()
        self.num_iter = num_iter
        self.num_steps = num_steps
        self.alpha_multiplier = alpha_init
        self.rho = rho_init
        self.rho_scale = rho_scale
        self.rho_max = rho_max
        self.weight_decay = weight_decay
        self.l1_coeff = l1_coeff
        self.h_tol = h_tol

    def validate_inputs(self):
        if isinstance(self.num_steps, list):
            if len(self.num_steps) < self.num_iter:
                missing_steps = self.num_iter - len(self.num_steps)
                self.vwarn(f"Using the first value from num_steps for the first {missing_steps} iterations")
                self.num_steps = missing_steps * [self.num_steps[0]] + self.num_steps
        elif isinstance(self.num_steps, (int, float)):
            self.num_steps = self.num_iter * [self.num_steps]
        else:
            self.vwarn("num_steps is None, will use default value from Unconstrained Solver")
            if self.unconstrained_solver.num_steps is None:
                raise ValueError("num_steps is None in Unconstrained Solver")
            else:
                self.num_steps = self.num_iter * [self.unconstrained_solver.num_steps]
        
        if self.l1_coeff > 0:
            # make sure the function l1_loss is implemented in the model
            if not hasattr(self.model, "l1_loss"):
                raise ValueError("Model does not have l1_loss method")

    def solve(self, dataset, model, unconstrained_solver, loss_fn, dag_fn):
        torch.set_default_dtype(self.dtype)
        self.model = model
        self.unconstrained_solver = unconstrained_solver
        self.vprint("Using Augmented Lagrangian Method to Solve the DAG constrained Problem")
        self.validate_inputs()

        h = float("inf")
        for i in tqdm(range(self.num_iter)):
            h_new = None
            while self.rho < self.rho_max:
                def augmented_loss(output, target):
                    # Original loss
                    loss = loss_fn(output, target)
                    
                    # L2 regularization
                    if self.weight_decay > 0:
                        l2_loss = 0.0
                        for param in model.parameters():
                            l2_loss += torch.sum(param**2)
                        loss += 0.5 * self.weight_decay * l2_loss
                        
                    if self.l1_coeff > 0:
                        loss += self.l1_coeff * model.l1_loss()
                        
                    # DAG constraint
                    h = dag_fn(model.adj())
                    
                    # Augmented Lagrangian terms
                    alm_term = self.alpha_multiplier * h + 0.5 * self.rho * h**2
                    
                    return loss + alm_term

                success = False
                model_copy = model.state_dict().copy()
                lr_scale = None
                lr_scheduler = False

                while not success:
                    self.vprint(
                        f"\nSolving ALM iteration #{i} with lambda={self.alpha_multiplier:.3f}, "
                        f"rho={self.rho:.3f} for {self.num_steps[i]} steps"
                    )
                    
                    unconstrained_solver.num_steps = self.num_steps[i]
                    success = unconstrained_solver(
                        dataset, model, augmented_loss, dag_fn, lr_scale, lr_scheduler
                    )

                    if not success:
                        model.load_state_dict(model_copy)
                        lr_scale = 0.5 if lr_scale is None else 0.5 * lr_scale
                        lr_scheduler = True
                        if lr_scale < 1e-6:
                            break
                
                with torch.no_grad():
                    h_new = dag_fn(model.adj()).item()
                if h_new > 0.25 * h:
                    self.rho *= self.rho_scale
                else:
                    break
            h = h_new
            self.vprint(f"Current h(W): {h:.4f}")
            self.alpha_multiplier += self.rho * h
            if h <= self.h_tol or self.rho >= self.rho_max:
                break



# TODO: Implement TOPOSolver (Bilevel)