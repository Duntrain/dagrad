from typing import Optional

import torch
import torch.optim as optim


class UnconstrainedSolver:
    def __init__(self):
        self.device = None
        self.dtype = None
        self.verbose = None
        self.suppress_warnings = None

    def __call__(self, *args, **kwds):
        return self.solve(*args, **kwds)

    def __repr__(self):
        return f"Unconstrained Solver: {self.__class__.__name__}"

    def solve(self):
        raise NotImplementedError


class GradientBasedSolver(UnconstrainedSolver):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_steps: Optional[int] = None,
        tol: Optional[float] = 1e-6,
        logging_steps: Optional[int] = 1e3,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.tol = tol
        self.lr = optimizer.param_groups[0]["lr"]
        self.logging_steps = logging_steps

    def solve(self, dataset, model, loss, dag_fn, lr_scale=None, lr_scheduler=None):
        torch.set_default_dtype(self.dtype)
        
        if lr_scale:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * lr_scale

        if lr_scheduler:
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

        def closure():
            self.optimizer.zero_grad()
            obj = loss(model(dataset), dataset)
            obj.backward()
            return obj

        obj_prev = 1e16
        for i in range(1, int(self.num_steps) + 1):
            with torch.no_grad():
                h = dag_fn(model.adj())
            if h.item() < 0:
                self.vprint(f"Found h negative {h.item()} at iter {i}. Stopping.")
                return False
            obj_new = self.optimizer.step(closure)
            if lr_scheduler and (i + 1) % 1000 == 0:  # every 1000 iters reduce lr
                scheduler.step()
            if i % self.logging_steps == 0 or i == self.num_steps:
                self.vprint(f"\n\tUnconstrained Solver iteration {i}")
                self.vprint(f"\t\th(W(model)): {h.item()}")
                self.vprint(f"\t\tscore(model): {obj_new.item()}")
                if abs((obj_prev - obj_new.item()) / obj_prev) <= self.tol:
                    break
                obj_prev = obj_new.item()
        return True
