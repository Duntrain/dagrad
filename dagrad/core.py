import numpy as np
from warnings import warn
from .utils.configure import METHODS, OPTIMIZERS, LOSS_FUNCTIONS, H_FUNCTIONS, REGULARIZERS
from .utils.configure import allowed_general_options, allowed_method_options, allowed_optimizer_options, loss_functions
from .utils.general_utils import validate_options, merge_dicts_if_needed,get_default, set_tuning_method_from_options, set_functions
from .utils.utils import is_dag, threshold_till_dag, simulate_dag, simulate_linear_sem, simulate_nonlinear_sem, set_random_seed, simulate_parameter, count_accuracy, generate_linear_data, generate_nonlinear_data
from .method.notears import notears_linear_numpy,notears_linear_torch, notears_nonlinear
from .method.dagma import dagma_linear_numpy, dagma_linear_torch,dagma_nonlinear
from .method.topo import topo_linear, topo_nonlinear

__all__ = ['dagrad','topo','notears','dagma']


def dagrad(X, 
            method='notears', 
            model=None, 
            loss_fn=None, 
            h_fn=None, 
            reg=None, 
            optimizer=None, 
            compute_lib=None, 
            device=None,
            general_options=None, 
            method_options=None, 
            optimizer_options=None):
    """
    This function learns the structure of a DAG from observational data using different methods.
    :math:`n`  number of samples, 
    :math:`p`  number of features.

    Parameters
    ----------
    X : numpy.array of shape :math:`(n, p)`
        The data matrix.

    method : str, default='notears'
        The method to learn the DAG. One of [:code:`'notears'`, :code:`'dagma'`, :code:`'topo'`].

    model : str, default=None
        The model of data generating process . One of [:code:`'linear'`, :code:`'nonlinear'`]. If None, it is set to :code:`'linear'`.

    loss_fn : str, default=None
        The loss function to use. One of [:code:`'l2'`, :code:`'logistic'`, :code:`'logll'`, :code:`'logdetll_ev'`, :code:`'logdetll_nv'` :code:`'user_loss'`]. If None, it is set to :code:`'l2'`.
        
        - For :code:`'user_loss'`, is not implemented yet, and it can be customized by the user.

    h_fn : str, default=None
        The function to compute the acyclicity constraint. One of [:code:`'h_exp_sq'`, :code:`'h_exp_abs'`, :code:`'h_poly_sq'`, :code:`'h_poly_abs'`, 
        :code:`'h_logdet_sq'`, :code:`'h_logdet_abs'`, :code:`'h_logdet_topo'`, :code:`'h_logdet_abs'`, :code:`'user_h'`]. If None, it is set to :code:`'h_exp_sq'`.

        - :code:`'abs'` stands for absolute value.
        - :code:`'sq'` stands for square.
        - :code:`'poly'` stands for polynomial.
        - :code:`'logdet'` stands for log determinant.
        - :code:`'topo'` is only specific to method :code:`'topo'`.
        - :code:`'user_h'` is not implemented yet, and it can be customized by the user.

    reg : str, default=None
        The regularizer to use. One of [:code:`'l1'`, :code:`'l2'`, :code:`'mcp'`, :code:`'none'`, :code:`'user_reg'`]. 
        If None, it is set to :code:`'l1'`.

        - :code:`'user_reg'` is not implemented yet, and it can be customized by the user.
        - :code:`'mcp'` stands for minimax concave penalty.
        - :code:`'none'` stands for no regularization.

    optimizer : str, default=None
        The optimizer to use. One of [:code:`'lbfgs'`, :code:`'adam'`, :code:`'sklearn'`]. If None, it is set to :code:`'lbfgs'`.

        - :code:`'sklearn'` is only used for method :code:`'topo'`.
        - Method :code:`'dagma'` is only compatible with the :code:`'adam'` optimizer.

    compute_lib : str, default=None
        The library to use for computation. One of [:code:`'numpy'`, :code:`'torch'`]. If None, it is set to :code:`'numpy'`.

        All nonlinear models use :code:`torch` for computation.
        Method :code:`'topo'` is only compatible with the :code:`'numpy'` compute_lib.

    device : str, default=None
        The device to use for computation. One of [:code:`'cpu'`, :code:`'cuda'`]. If None, it is set to :code:`'cpu'`.

    general_options : dict, default=None
        General options for the method. If None, it is set to default values. Please refer to :ref:`options` for more details.

        Allowed options are:
        
        - :code:`'lambda1'`: float, default=0.1
            Weight for l1 penalty.
        - :code:`'lambda2'`: float, default=0.01
            Weight for l2 penalty.
        - :code:`'gamma'`: float, default=1.0
            Hyperparameter for (quasi-)MCP.
        - :code:`'w_threshold'`: float, default=0.3
            Threshold for the output adjacency matrix.
        - :code:`'initialization'`: None or np.ndarray or torch.Tensor or nn.Module, default=None
            Initialization for the parameters of the model.
        - :code:`'tuning_method'`: str or None, default=None
            Method for tuning hyperparameters. One of :code:`'cv'`, :code:`'decay'`.
        - :code:`'K'`: int, default=5
            Number of folds for cross-validation.
        - :code:`'reg_paras'`: list, default=None
            List of regularization parameters.
        - :code:`'user_params'`: Any, default=None
            User-defined parameters.

    method_options : dict, default=None
        Method-specific options for the method. If None, it is set to default values.

        More options are available for each method. Please refer to :ref:`options` for more details, 
        or check :code:`allowed_method_options` in :code:`dagrad/utils/configure.py`.

    optimizer_options : dict, default=None
        Optimizer-specific options for the optimizer. If None, it is set to default values.

        More options are available for each optimizer. Please refer to :ref:`options` for more details, 
        or check :code:`allowed_optimizer_options` in :code:`dagrad/utils/configure.py`.

    Returns
    -------
    W_est : array-like of shape :math:`(p, p)`
        The estimated adjacency matrix of the DAG.
    """

    # get default values
    model, loss_fn, h_fn, reg, optimizer, compute_lib, device = get_default(model, method, loss_fn, h_fn, reg, optimizer, compute_lib, device)

    # check if inputs are valid
    assert method in METHODS, f"method must be one of {METHODS}"
    assert model in ['linear', 'nonlinear'], "model must be one of ['linear', 'nonlinear']"
    assert loss_fn in LOSS_FUNCTIONS, f"loss_fn must be one of {LOSS_FUNCTIONS}"
    assert h_fn in H_FUNCTIONS, f"h_fn must be one of {H_FUNCTIONS}"
    assert optimizer in OPTIMIZERS, f"optimizer must be one of {OPTIMIZERS}"
    assert reg in REGULARIZERS, f"regularizer must be one of {REGULARIZERS}"
    assert compute_lib in ['numpy', 'torch'], "compute_lib must be one of ['numpy', 'torch']"
    assert device in ['cpu', 'cuda'], "device must be one of ['cpu', 'cuda']"

    # check if options are valid
    if general_options is None:
        general_options = {}
    else:
        validate_options(general_options, allowed_general_options)

    if method_options is None:
        method_options = {}
    else:
        validate_options(method_options, allowed_method_options[model][method]) 

    if optimizer_options is None:
        optimizer_options = {}
    else:
        validate_options(optimizer_options, merge_dicts_if_needed(allowed_optimizer_options[compute_lib][optimizer])) 

    # check if the options are compatible with the method
    if method == 'dagma' and not (h_fn in ['h_logdet_sq','h_logdet_abs', 'user_h']):
        warn("DAGMA is framework using logdet acyclicity constraint. Changing h_fn to h_logdet_sq")
        h_fn = 'h_logdet_sq'
    if method == 'dagma' and optimizer == 'lbfgs':
        warn("DAGMA is not compatible with lbfgs optimizer for now[ADD LBFGS in future]. Changing optimizer to adam")
        optimizer = 'adam'
    if method == 'notears' and h_fn == 'h_logdet_sq':
        warn("Logdet acyclicity constraint is restricted to DAGMA. Changing h_fn to h_exp_sq")
        h_fn = 'h_exp_sq'
    if method == 'topo' and 'topo' not in h_fn:
        warn("TOPO has its own acyclicity constraint. Changing h_fn to h_logdet_topo")
        h_fn = 'h_logdet_topo'
    if method == 'topo' and model == 'linear' and (device == 'cuda' or compute_lib =='torch'):
        raise NotImplementedError("For Linear Model, TOPO is not good for PyTorch, please use compute_lib = 'numpy' and device = 'cpu'.")
    if optimizer == 'sklearn' and method!= 'topo':
        raise ValueError("sklearn is only used for TOPO method.")

    compute_lib = 'torch' if device == 'cuda' else compute_lib

    if method == 'topo' and (loss_fn in ['l2', 'logsitc']) and optimizer != 'sklearn' and model == 'linear':
        warn(f"For l2 or logistic loss, TOPO would be faster using optimizer: sklearn. But you are currently using optimizer: {optimizer}.")
    

    options = {}
    options.update(general_options)
    options.update(method_options)
    options.update(optimizer_options)
    if model =='nonlinear' and 'tuning_method' in options and options['tuning_method'] is not None:
        raise ValueError("Parameter tuning is not implemented for nonlinear models, since it is computational intense.")
    if model == 'linear':
        if compute_lib == 'numpy':
            if method == 'notears':
                # if the setting corresponds to the original notears implementation, use the original implementation
                if loss_fn in ['logistic', 'l2'] and h_fn =='h_exp_sq' and reg == 'l1' and optimizer == 'lbfgs' and device == 'cpu' and ('tuning_method' not in options or options['tuning_method'] is None):
                    from notears.linear import notears_linear
                    print('Using Original NOTEARS Implementation for Linear Model')
                    W_est = notears_linear(X = X, 
                                           lambda1 = options.get('lambda1', 0.1), 
                                           loss_type = loss_fn, 
                                            max_iter = options.get('main_iter', 100), 
                                            h_tol = options.get('h_tol', 1e-8), 
                                            rho_max = options.get('rho_max', 1e+16), 
                                            w_threshold = options.get('w_threshold', 0.3))
                else:
                    if 'tuning_method' in options and options['tuning_method'] is not None:
                        W_est = parameter_tuning(dag_learning_func = notears_linear_numpy, X = X, model = model, loss_fn = loss_fn, h_fn = h_fn, reg = reg, optimizer = optimizer, options = options)
                    else:
                        W_est = notears_linear_numpy(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                            reg = reg, optimizer = optimizer, **options)
            elif method == 'dagma':
                # if the setting corresponds to the original dagma implementation, use the original implementation
                if loss_fn in ['logistic', 'l2'] and h_fn =='h_logdet_sq' and reg == 'l1' and optimizer == 'adam' and device == 'cpu' and ('tuning_method' not in options or options['tuning_method'] is None):
                    from dagma.linear import DagmaLinear
                    print('Using Original DAGMA Implementation for Linear Model')
                    model = DagmaLinear(loss_type = loss_fn, verbose=options.get('verbose', False), dtype=options.get('dtype', np.float64))
                    W_est = model.fit(X = X, lambda1 = options.get('lambda1', 0.03), 
                                      w_threshold=options.get('w_threshold', 0.3), 
                                      T = options.get('T',5), 
                                      mu_init = options.get('mu_init',1.0), 
                                      mu_factor = options.get('mu_factor', 0.1), 
                                      s = options.get('s', [1.0, .9, .8, .7, .6]),
                                      warm_iter = options.get('warm_iter',3e4), 
                                      max_iter = options.get('main_iter', 6e4), 
                                      lr = options.get('lr', 0.0003), 
                                      checkpoint= options.get('check_iterate', 1000),
                                      beta_1= options.get('betas',(0.9,0.999))[0], 
                                      beta_2= options.get('betas',(0.9,0.999))[1],
                                      exclude_edges=options.get('exclude_edges', None), 
                                      include_edges=options.get('include_edges', None),
                        )
                else:
                    if 'tuning_method' in options and options['tuning_method'] is not None:
                        W_est = parameter_tuning(dag_learning_func = dagma_linear_numpy, X = X, model = model, loss_fn = loss_fn, h_fn = h_fn, reg = reg, optimizer = optimizer, options = options)
                    else:
                        W_est = dagma_linear_numpy(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                            reg = reg, optimizer = optimizer, **options)
            elif method == 'topo':
                # the original implementation of TOPO is same as current implementation, so use the current implementation
                if 'tuning_method' in options and options['tuning_method'] is not None:
                    W_est = parameter_tuning(dag_learning_func = topo_linear, X = X, model = model, loss_fn = loss_fn, h_fn = h_fn, reg = reg, optimizer = optimizer, options = options)
                else:
                    W_est = topo_linear(X = X, loss_fn = loss_fn,  h_fn = h_fn, reg = reg, optimizer = optimizer, **options)

        elif compute_lib == 'torch':
            options['device'] = device
            if method == 'notears':
                if 'tuning_method' in options and options['tuning_method'] is not None:
                    W_est = parameter_tuning(dag_learning_func = notears_linear_torch, X = X, model = model, loss_fn = loss_fn, h_fn = h_fn, reg = reg, optimizer = optimizer, options = options)
                else:
                    W_est = notears_linear_torch(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                        reg = reg, optimizer = optimizer, **options)
            elif method == 'topo':
                raise NotImplementedError("For Linear Model, Torch is not implemented for method 'topo', please use 'numpy' as computation library.")
            elif method == 'dagma':
                if 'tuning_method' in options and options['tuning_method'] is not None:
                    W_est = parameter_tuning(dag_learning_func = dagma_linear_torch, X = X, model = model, loss_fn = loss_fn, h_fn = h_fn, reg = reg, optimizer = optimizer, options = options)
                else:
                    W_est = dagma_linear_torch(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                        reg = reg, optimizer = optimizer, **options)
    elif model == 'nonlinear':
        options['device'] = device
        
        if method =='notears':
            if loss_fn == 'l2' and h_fn == 'h_exp_sq' and 'reg'=='l1' and optimizer =='lbfgs' and device =='cpu':
                print('Using Original NOTEARS Implementation for Nonlinear Model')
                from notears.nonlinear import notears_nonlinear as notears_nonlinear_original
                from notears.nonlinear import NotearsMLP
                model = NotearsMLP(dims = [X.shape[1], 40, 1], 
                                   bias= options.get('bias',True))
                W_est = notears_nonlinear_original(model = model,
                                          X = X,
                                          lambda1 = options.get('lambda1', 0.01),
                                          lambda2= options.get('lambda2', 0.01),
                                          max_iter= options.get('main_iter', 100),
                                          h_tol= options.get('h_tol', 1e-8),
                                          rho_max= options.get('rho_max', 1e+16),
                                          w_threshold= options.get('w_threshold', 0.3),
                                        )
            else:
                W_est = notears_nonlinear(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                            reg = reg, optimizer = optimizer, **options)
                
        
        elif method == 'dagma':
            if loss_fn =='logll' and h_fn == 'h_logdet_sq' and 'reg'=='l1' and optimizer =='adam' and device =='cpu':
                print('Using Original DAGMA Implementation for Nonlinear Model')
                from dagma.nonlinear import DagmaNonlinear, DagmaMLP
                import torch
                model = DagmaNonlinear(DagmaMLP(dims = [X.shape[1], 10, 1], 
                                        bias= options.get('bias',True),
                                        dtype=options.get('dtype', torch.double)),
                                        verbose= options.get('verbose', False),
                                        dtype=options.get('dtype', torch.double)
                                        )
                W_est = model.fit(X = X, 
                                  lambda1 = options.get('lambda1', 0.02), 
                                  lambda2 = options.get('lambda2', 0.005),
                                  T = options.get('T', 4),
                                  mu_init= options.get('mu_init', 0.1),
                                  mu_factor= options.get('mu_factor', 0.1),
                                  s = options.get('s',1.0),
                                  warm_iter= options.get('warm_iter', 5e4),
                                  main_iter= options.get('main_iter', 8e4),
                                  lr = options.get('lr', 0.0002),
                                  w_threshold= options.get('w_threshold', 0.3),
                                  checkpoint= options.get('check_iterate', 1000),
                    )
            else:
                W_est = dagma_nonlinear(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                            reg = reg, optimizer = optimizer, **options)
        elif method =='topo':
            W_est = topo_nonlinear(X = X, loss_fn = loss_fn, h_fn = h_fn, 
                                    reg = reg, optimizer = optimizer, **options)
        
    return W_est

def parameter_tuning(dag_learning_func, X, model, loss_fn, h_fn, reg, optimizer, options):

    tuning_method, K, reg_paras, initialization = set_tuning_method_from_options(options = options, reg = reg)
    assert tuning_method in ['cv','decay'], "Tuning method must be one of ['cv','decay']"
    
    def cross_validation(K, reg_paras, W_init = None):
        _loss = set_functions(loss_fn, loss_functions)
        n,d = X.shape
        indices =  np.arange(n)
        fold_size = n//K
        folds = []
        for k in range(K):
            if k == K-1:
                folds.append(indices[k*fold_size:])
            else:
                folds.append(indices[k*fold_size:(k+1)*fold_size])
        Err_list = []
        # Here we need to consider whether we are using torch or numpy, left for future implementation
        if reg == 'l1' or reg == 'l2':
            assert isinstance(reg_paras[0],(float,int)), "For l1 or l2 regularization, only one parameter is needed."
            lambda1s = reg_paras
            print(f"Working with {K} fold cross validation on {reg} with lambda1s: {lambda1s}")
            for lambda1 in lambda1s:
                Err_total = 0
                for k in range(K):
                    test_indices = folds[k]
                    train_indices = np.concatenate([folds[j] for j in range(K) if j != k])
                    X_train = X[train_indices]
                    X_test = X[test_indices]
                    W_est = dag_learning_func(X = X_train, 
                                            loss_fn = loss_fn, 
                                            h_fn = h_fn, 
                                            reg = reg, 
                                            optimizer = optimizer, **options, **{'lambda1':lambda1, 'w_threshold':0.0, 'initialization': W_init})
                    
                    Err, _ = _loss(W = W_est, X = X_test)
                    Err_total += Err
                Err_average = Err_total / K
                Err_list.append(Err_average)
            opt_lambda= lambda1s[Err_list.index(min(Err_list))]
            print(f"Optimal lambda1: {opt_lambda}")
            print(f"Retrain the model with optimal lambda1")
            # retrain the model 
            W_est = dag_learning_func(X = X, 
                                            loss_fn = loss_fn, 
                                            h_fn = h_fn, 
                                            reg = reg, 
                                            optimizer = optimizer, **options, **{'lambda1':opt_lambda, 'initialization': W_init})
            print(f"Model retrained with optimal lambda1")
            if not is_dag(W_est):
                W_est = threshold_till_dag(W_est)
            return W_est
        
        elif reg =='mcp':
            
            assert len(reg_paras)==2 and isinstance(reg_paras[0][0],(float,int)), "For mcp regularization, two parameters are needed. First parameter is lambda1 and second parameter is gamma."
            lambda1s, gammas = reg_paras[0], reg_paras[1]
            print(f"Working with {K} fold cross validation on {reg} with gammas: {gammas} and lambda1s: {lambda1s}")
            reg_space = [[gamma, lambda1] for gamma in gammas for lambda1 in lambda1s]
            params_list = []
            for gamma, lambda1 in reg_space:
                Err_total = 0
                for k in range(K):
                    test_indices = folds[k]
                    train_indices = np.concatenate([folds[j] for j in range(K) if j != k])
                    X_train = X[train_indices]
                    X_test = X[test_indices]
                    # fit model
                    W_est = dag_learning_func(X = X_train, 
                                            loss_fn = loss_fn, 
                                            h_fn = h_fn, 
                                            reg = reg, 
                                            optimizer = optimizer, **options, **{'lambda1':lambda1, 'w_threshold':0.0, 'gamma':gamma, 'initialization': W_init})
                    # evaluate model
                    Err, _ = _loss(W = W_est, X = X_test)
                    #Err = 0.5 * ((X_test - X_test@W_est)**2).sum()
                    Err_total += Err
                Err_average = Err_total / K
                Err_list.append(Err_average)
                params_list.append([gamma, lambda1])
            min_index = np.argmin(Err_list)
            opt_gamma, opt_lambda = opt_gamma, opt_lambda1 = params_list[min_index]
            print(f"Optimal gamma: {opt_gamma}, Optimal lambda1: {opt_lambda1}")
            print(f"Retrain the model with optimal gamma and lambda1")
            # retrain the model
            W_est = dag_learning_func(X = X, 
                                            loss_fn = loss_fn, 
                                            h_fn = h_fn, 
                                            reg = reg, 
                                            optimizer = optimizer, **options, **{'lambda1':opt_lambda, 'gamma':opt_gamma, 'initialization': W_init})
            print(f"Model retrained with optimal gamma and lambda1")
            if not is_dag(W_est):
                W_est = threshold_till_dag(W_est)
            return W_est
        else:
            return NotImplementedError("Parameter tuning is not implemented for user defined regularizer.")
        
        
    def decay(reg_paras, W_init = None):
        _loss = set_functions(loss_fn, loss_functions)
        Err_list = [float('inf')]
        if reg == 'l1' or reg == 'l2':
            assert isinstance(reg_paras[0],(float,int)), "For l1 or l2 regularization, only one parameter is needed."
            lambda1s = reg_paras.copy()
            # we need to sort the lambda1s in descending order, we start with the largest lambda1
            lambda1s.sort(reverse = True)
            if W_init is None:
                W_est = None
            else:
                W_est = W_init.copy()

            optimal_lambda1 = lambda1s[-1]
            print(f"Working with decay on {reg} with lambda1s: {lambda1s}")
            for i, lambda1 in enumerate(lambda1s):
                # fit the model
                W_est = dag_learning_func(X = X, 
                                        loss_fn = loss_fn, 
                                        h_fn = h_fn, 
                                        reg = reg, 
                                        optimizer = optimizer, **options, **{'lambda1':lambda1, 'w_threshold':0.0, 'initialization': W_est})

                # evaluate model
                Err, _ = _loss(W = W_est, X = X)
                # Err = 0.5/ X.shape[0] * ((X - X@W_est)**2).sum()
                Err_list.append(Err)
                if Err_list[-1]> Err_list[-2]: # current loss is greater than previous loss
                    optimal_lambda1 = lambda1s[i-1]
                    break
            print(f"Optimal lambda1: {optimal_lambda1}")
            print(f"Retrain the model with optimal lambda1")
            # retrain the model
            W_est = dag_learning_func(X = X, 
                                    loss_fn = loss_fn, 
                                    h_fn = h_fn, 
                                    reg = reg, 
                                    optimizer = optimizer, **options, **{'lambda1':optimal_lambda1, 'initialization': W_init})
            print(f"Model retrained with optimal lambda1")
            if not is_dag(W_est):
                W_est = threshold_till_dag(W_est)
            return W_est
        elif reg =='mcp':
            assert len(reg_paras)==2 and isinstance(reg_paras[0][0],(float,int)), "For mcp regularization, two parameters are needed. First parameter is lambda1 and second parameter is gamma."
            assert len(reg_paras[0])==len(reg_paras[1]), "The number of lambda1s and gammas must be the same."
            reg_paras_= reg_paras.copy()
            lambda1s, gammas = reg_paras_[0],reg_paras_[1]
            lambda1s.sort(reverse = True)
            gammas.sort(reverse = True)
            if W_init is None:
                W_est = None
            else:
                W_est = W_init.copy()
            
            optimal_gamma, optimal_lambda1 = gammas[-1], lambda1s[-1]
            print(f"Working with decay on {reg} with gammas: {gammas} and lambda1s: {lambda1s}")
            for i, (gamma, lambda1) in enumerate(zip(gammas, lambda1s)):
                W_est = dag_learning_func(X = X, 
                                        loss_fn = loss_fn, 
                                        h_fn = h_fn, 
                                        reg = reg, 
                                        optimizer = optimizer, **options, **{'lambda1':lambda1, 'gamma':gamma, 'w_threshold':0.0, 'initialization': W_est})
                Err, _ = _loss(W = W_est, X = X)
                Err_list.append(Err)
                if Err_list[-1]> Err_list[-2]:
                    optimal_gamma, optimal_lambda1 = gammas[i-1], lambda1s[i-1]
                    break
            print(f"Optimal gamma: {optimal_gamma}, Optimal lambda1: {optimal_lambda1}")
            print(f"Retrain the model with optimal gamma and lambda1")

            # retrain the model
            W_est = dag_learning_func(X = X, 
                                    loss_fn = loss_fn, 
                                    h_fn = h_fn, 
                                    reg = reg, 
                                    optimizer = optimizer, **options, **{'lambda1':optimal_lambda1, 'gamma':optimal_gamma, 'initialization': W_init})
            print(f"Model retrained with optimal gamma and lambda1")
            if not is_dag(W_est):
                W_est = threshold_till_dag(W_est)
            return W_est
        else:
            return NotImplementedError("Parameter tuning is not implemented for user defined regularizer.")



    if model =='linear':
        if tuning_method == 'cv':
            return cross_validation(K, reg_paras, W_init = initialization)
        elif tuning_method == 'decay':
            return decay(reg_paras, W_init = initialization)
        else:
            raise NotImplementedError("Parameter tuning is not implemented for user defined regularizer.")
    else:
        raise NotImplementedError("Parameter tuning is not implemented for nonlinear model.")




def notears(X, model = 'linear'):
    '''
    This is a wrapper function for dagrad with method notears, it calls the original notears implementation

    Parameters
    ----------
    X : numpy.array of shape :math:`(n, p)`
        The data matrix.
    model : str, default='linear'
        The model of data generating process . 

    Returns
    -------
    W_est : numpy.array of shape :math:`(p, p)`
        The estimated adjacency matrix of the DAG.
    '''
    W_est = dagrad(X = X, method= 'notears', model = model)
    return W_est

def dagma(X, model = 'linear'):
    '''
    This is a wrapper function for dagrad with method dagma, it calls the original dagma implementation

    Parameters
    ----------
    X : numpy.array of shape :math:`(n, p)`
        The data matrix.
    model : str, default='linear'
        The model of data generating process .
        
    Returns
    -------
    W_est : numpy.array of shape :math:`(p, p)`
        The estimated adjacency matrix of the DAG.
    '''
    W_est = dagrad(X = X, method= 'dagma', model = model)
    return W_est

def topo(X, model = 'linear'):
    '''
    This is a wrapper function for dagrad with method topo

    Parameters
    ----------
    X : numpy.array of shape :math:`(n, p)`
        The data matrix.
    model : str, default='linear'
        The model of data generating process .

    Returns
    -------
    W_est : numpy.array of shape :math:`(p, p)`
        The estimated adjacency matrix of the DAG.
    '''
    W_est = dagrad(X = X, method= 'topo', model = model)
    return W_est


if __name__ == "__main__":
    set_random_seed(2)
    n, d, s0, graph_type, sem_type = 1000, 10, 10, 'ER', 'gauss'
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,sem_type) # Generate the data

    model = 'linear' # Define the model
    W_notears = dagrad(X, model = model, method = 'notears') # Learn the structure of the DAG using Notears
    W_dagma = dagrad(X, model = model, method = 'dagma') # Learn the structure of the DAG using Dagma
    W_topo = dagrad(X, model = model, method = 'topo') # Learn the structure of the DAG using Topo
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, sem type: {sem_type}")

    acc_notears = count_accuracy(B_true, W_notears != 0) # Measure the accuracy of the learned structure using Notears
    print('Accuracy of Notears:', acc_notears)

    acc_dagma = count_accuracy(B_true, W_dagma != 0) # Measure the accuracy of the learned structure using Dagma
    print('Accuracy of Dagma:', acc_dagma)

    acc_topo = count_accuracy(B_true, W_topo != 0) # Measure the accuracy of the learned structure using Topo
    print('Accuracy of Topo:', acc_topo)

    # test cross validation for linear model with method notears
    # W_est = dagrad(X = X, method = 'notears', reg = 'l1', general_options = {'tuning_method':'cv'})
    # acc = count_accuracy(B_true, W_est != 0)
    # print(acc)


    

    # Uncomment the following code to test for nonlinear model
    # # test for nonlinear 
    # from utils import utils
    # utils.set_random_seed(1234)
    # n, d, s0 = 1000, 5, 10
    # graph_type, sem_type = 'ER', 'mlp'
    # B_true = utils.simulate_dag(d, s0, graph_type)
    # W_true = utils.simulate_parameter(B_true)
    # X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
    
    # # test for nonlinear model with method notears with default options
    # W_est = dagrad(X = X, method = 'notears', model = 'nonlinear')

    # # test for nonlinear model with method dagma with default options
    # W_est = dagrad(X = X, method = 'dagma', model = 'nonlinear')

    # # test for nonlinear model with method topo with default options
    # W_est = dagrad(X = X, method = 'topo', model = 'nonlinear')

    # acc = utils.count_accuracy(B_true, W_est != 0)
    # print(acc)


    