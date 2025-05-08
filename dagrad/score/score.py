import numpy as np
import torch
from scipy.special import expit as sigmoid

class reg_fn:
    @staticmethod
    def user_reg(W, **kwargs):
        """
        This is a user-defined regularization function. Users can define their own regularization function by customizing this function.

        Parameters
        ----------
        W : numpy array or torch tensor
            Weight matrix.

        Returns
        -------
        reg_val : float
            Value of the regularization function.
        reg_grad : numpy array or torch tensor
            Gradient of the regularization function.

        Important
        ---------
        If users work with **numpy** as the computation library, the output of the ``user_reg`` function should be 
        (value of regularization, gradient of regularization) in numpy arrays.
        If users work with **torch** as the computation library, the output of the ``user_reg`` function should be 
        the value of regularization in a torch tensor.
        """
        
        if isinstance(W, np.ndarray):
            pass
        elif isinstance(W, torch.Tensor):
            pass
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        raise NotImplementedError("User-defined regularization is not implemented yet. User are free to define their own regularization function")


    @staticmethod
    def no_reg(W, **kwargs):
        
        if isinstance(W, np.ndarray):
            return 0, np.zeros_like(W)
        elif isinstance(W, torch.Tensor):
            return 0
        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def l1_reg(W,**kwargs):
        
        lambda1 = kwargs.get('lambda1',0.1)
        if isinstance(W, np.ndarray):
            return lambda1*np.abs(W).sum(), lambda1*np.sign(W)
        elif isinstance(W, torch.Tensor):
            return lambda1*torch.abs(W).sum()
        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def l2_reg(W,**kwargs):
        
        lambda1 = kwargs.get('lambda1',0.1)
        if isinstance(W, np.ndarray):
            return lambda1/2 * (W ** 2).sum(), lambda1*W
        elif isinstance(W, torch.Tensor):
            return lambda1/2 * torch.sum(W**2)
        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def mcp_reg(W, **kwargs):

        lambda1 = kwargs.get('lambda1', 0.1)
        gamma = kwargs.get('gamma',1)
        if isinstance(W, np.ndarray):
            cond = np.abs(W) <= gamma 
            reg_val = lambda1* np.where(cond, np.abs(W) - W**2/(2*gamma) , gamma/2)
            reg_grad = lambda1 * np.where(cond, np.sign(W) -W / gamma, 0)
            return reg_val.sum(), reg_grad
        elif isinstance(W, torch.Tensor):
            cond = torch.abs(W) <= gamma
            reg_val = lambda1* torch.where(cond, torch.abs(W) - W**2/(2*gamma) , gamma/2)
            reg_grad = lambda1 * torch.where(cond, torch.sign(W) -W / gamma, 0)
            return reg_val.sum()
        else:
            raise ValueError("W must be either numpy array or torch tensor")

class loss_fn:
    @staticmethod
    def user_loss(W,X,**kwargs):
        """
        This is a user-defined loss function for **linear** model. Users can define their own loss function by customizing this function.

        Parameters
        ----------
        W : numpy array or torch tensor
            Weight matrix.
        X : numpy array or torch tensor
            Data matrix.

        Returns
        -------
        loss : float
            Value of the loss function.
        G_loss : numpy array or torch tensor
            Gradient of the loss function.

        Important
        ---------
        If users work with **numpy** as the computation library, the output of the ``user_loss`` function should be 
        (value of loss, gradient of loss) in numpy arrays.
        If users work with **torch** as the computation library, the output of the ``user_loss`` function should be 
        the value of the loss in a torch tensor.

        """
        
        if isinstance(W, np.ndarray):
            pass
        elif isinstance(W, torch.Tensor):
            pass
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        raise NotImplementedError("User-defined loss is not implemented yet. User are free to define their own loss function")



    @staticmethod
    def l2_loss(W, X, **kwargs):
        
        if isinstance(W, np.ndarray):
            M = X @ W
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
            return loss, G_loss
        elif isinstance(W, torch.Tensor):
            M = X.matmul(W)
            R = X - M
            loss = 0.5 / X.shape[0] * torch.sum(R ** 2)
            return loss
        else:
            raise ValueError("W must be either numpy array or torch tensor")
    @staticmethod
    def logistic_loss(W,X,**kwargs):
        """
        Calculate the logistic loss, and correpsonding gradient of the linear model
        
        Structural Equation Model: x_i = Bernoulli(sigmoid(\sum_j w_ji x_j))

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)

        Returns:
        loss: float
        gradient: ndarray, shape (n_nodes, n_nodes)
        """
        if isinstance(W, np.ndarray):
            M = X @ W
            loss = 1.0 / X.shape[0] * (np.logaddexp(0,M)-X * M).sum()
            G_loss =  1.0 / X.shape[0] * X.T @ (sigmoid(M)- X)
            return loss, G_loss
        elif isinstance(W, torch.Tensor):
            M = X.matmul(W)
            loss = 1.0 / X.shape[0] * torch.sum(torch.logaddexp(torch.zeros_like(M), M) - X * M)
            return loss
        else:
            raise ValueError("W must be either numpy array or torch tensor")
    
    @staticmethod
    def logll_loss(W,X,**kwargs): 
        """
        Calculate the loglikelihood loss, and correpsonding gradient of the linear model
        
        Structural Equation Model: x_i = (\sum_j w_ji x_j) + \epsilon where \epsilon is normal distribution

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)
        """
        if isinstance(W, np.ndarray):
            mean = np.mean(X, axis=0)
            X = X - mean
            n, d = X.shape
            Sigma = 1/n * X.T@X
            I = np.eye(d)
            diag_elements = np.diag((W-I).transpose()@Sigma@ (W-I))
            loss = 1/2* np.sum(np.log(diag_elements))
            G_loss = Sigma@(W-I)@np.diag(1/diag_elements)
            return loss, G_loss
        elif isinstance(W, torch.Tensor):
            n, d = X.shape
            X = X - torch.mean(X, dim=0)
            Sigma = 1/n * X.T@X
            I = torch.eye(d)
            diag_elements = torch.diag((W-I).transpose(0,1)@Sigma@ (W-I))
            loss = 1/2* torch.sum(torch.log(diag_elements))
            return loss
            
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        
    @staticmethod
    def logdetll_ev_loss(W, X, **kwargs):
        """
        Calculate the logdet loglikelihood loss assuming equal noise variance

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)
        """
        if isinstance(W, np.ndarray):
            raise ValueError("logdetll_ev_loss is not implemented for numpy")
        elif isinstance(W, torch.Tensor):
            return (
                0.5
                * W.shape[0]
                * torch.log(torch.square(torch.linalg.matrix_norm(X - X @ W)))
                - torch.linalg.slogdet(torch.eye(W.shape[1]) - W)[1]
            )

        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def logdetll_nv_loss(W, X, **kwargs):
        """
        Calculate the logdet loglikelihood loss assuming non-equal noise variance

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)
        """
        if isinstance(W, np.ndarray):
            raise ValueError("logdetll_nv_loss is not implemented for numpy")
        elif isinstance(W, torch.Tensor):
            return (
                0.5 * torch.sum(torch.log(torch.sum(torch.square(X - X @ W), axis=0)))
                - torch.linalg.slogdet(torch.eye(W.shape[1]) - W)[1]
            )

        else:
            raise ValueError("W must be either numpy array or torch tensor")


class nl_loss_fn:
    @staticmethod
    def user_nl_loss(output, target, **kwargs):
        """
        This is a user-defined loss function for **nonlinear** model. Users can define their own loss function by customizing this function.

        Parameters
        ----------
        output : torch tensor
            Output of the model.
        target : torch tensor
            Target value.

        Returns
        -------
        loss : torch tensor
            Value of the loss function.

        Important
        ---------
        The output of the ``user_nl_loss`` function should be the value of the loss in a torch tensor.
        """


        raise NotImplementedError("User defined loss is not implemented yet. User are free to define their own loss function.")
    
    @staticmethod
    def nl_l2_loss(output, target, **kwargs):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    @staticmethod
    def nl_logistic_loss(output,target, **kwargs):
        import torch.nn as nn
        bce_loss = nn.BCELoss()
        return bce_loss(output, target)
        
    
    @staticmethod
    def nl_logll_loss(output, target, **kwargs): 
        n, d = target.shape
        loss = 0.5 * d * torch.log(1/n* torch.sum((output - target)**2))
        return loss