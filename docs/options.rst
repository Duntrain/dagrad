.. _options:

Options
========================

In this section, we will discuss the all the options that can be specified for the method and the optimizer.

General Options
^^^^^^^^^^^^^^^^

The following options are available for all methods. Its specification can be done through the ``general_options = {}``. The examples are given below.

.. list-table:: Allowed General Options
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``'lambda1'``
     - float, int
     - Weight for :math:`\ell_1` penalty
   * - ``'lambda2'``
     - float, int
     - Weight for :math:`\ell_2` penalty
   * - ``'gamma'``
     - float, int
     - Hyperparameter for quasi-MCP
   * - ``'w_threshold'``
     - float, int
     - Threshold for the output adjacency matrix
   * - ``'initialization'``
     - None
        
       np.ndarray 
     
       torch.Tensor
       
       nn.Module
     - Initialization for model parameters
   * - ``'turning_method'``
     - str, None
     - Turning method for regularization parameters. 
     
       The options are [``'cv'``, ``'decay'``].

       ``None``: no tuning is applied.

       ``'cv'``: Cross-validation

       ``'decay'``: Decay search defined in [1]_

   * - ``'K'``
     - int
     - Number of folds for cross-validation
   * - ``'reg_paras'``
     - List
     - Regularization parameters for the turning method. 
   
   * - ``'user_params'``
     - Any
     - User-defined parameters, which will be passed to the 
     
       following functions: ``user_h``, ``user_reg``, and ``user_loss``.

       These parameters allow users to customize the behavior of their 
       
       loss, regularization, and acyclicity constraint functions.
.. code-block:: python

    from dagrad import learn, generate_linear_data
    import numpy as np
    # Define the parameters of the data
    n, d, s0, graph_type, noise_type = 1000, 10, 10, 'ER', 'gauss'
    
    # Generate the data
    X, W_true, B_true = generate_linear_data(n, d, s0, graph_type, noise_type)
    
    # Define the model
    model = 'linear'
    
    # Define the initializations
    initialization = np.random.randn(d, d)
    np.fill_diagonal(initialization,0)

    # Define the general options
    general_options = {
        'lambda1': 0.1,
        'lambda2': 0.1,
        'gamma': 0.1,
        'w_threshold': 0.3,
        'initialization': initialization
        'user_params': {'param1': 1, 'param2': 2}
    }
    
    # Learn the DAG
    W_est = learn(X, method='notears', general_options=general_options)

    # Use cross-validation to tune the default regularization parameters with l1 penalty
    reg = 'l1'
    general_options = {
        'w_threshold': 0.3,
        'initialization': initialization
        'turning_method': 'cv', # or 'decay'
    }
    W_est = learn(X, method='notears', reg = reg, general_options=general_options)

    # Or specify the regularization parameters with l1 penalty and number of folds by yourself
    general_options = {
        'w_threshold': 0.3,
        'initialization': initialization
        'turning_method': 'cv', # or 'decay'
        'K': 5,
        'reg_paras': [0, 0.1, 0.2, 0.3, 0.4, 0.5] #
    }

    W_est = learn(X, method='notears',reg = reg, general_options=general_options)

    # Or MCP penalty with hyperparameters lambda1, gamma and use decay search for faster computation
    reg = 'mcp'
    general_options = {
        'w_threshold': 0.3,
        'initialization': initialization
        'turning_method': 'cv',
        'reg_paras': [[0, 0.05, 0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5, 1]] # first is lambda1s, and second is gammas
    }

    W_est = learn(X, method='notears', reg = reg, general_options=general_options)

Method Options
^^^^^^^^^^^^^^^^


The following options are available for different methods under :code:`'linear'` and :code:`''nonlinear'` categories. Its specification can be done through the ``method_options = {}``. The examples are given below.


.. list-table:: Method Options
   :header-rows: 1
   :widths: 20 20 40

   * - ``model``
     - ``method``
     - Parameter
   * - ``'linear'``
     - ``'notears'``
     - 
         - rho_max: int
         - main_iter: int
         - rho: float
         - h_tol: float
         - dtype: type
         - verbose: bool
   * - 
     - ``'dagma'``
     - 
         - T: int
         - mu_init: float
         - mu_factor: float
         - s: typing.Union[typing.List[float], float]
         - warm_iter: int
         - main_iter: int
         - dtype: type
         - exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] 
         
           or  None
         - include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] 
         
           or None
         - verbose: bool
   * - 
     - ``'topo'``
     - 
         - no_large_search: int
         - size_small: int
         - size_large: int
         - topo: (list, type(None))
         - dtype: type
         - verbose: bool
   * - ``nonlinear``
     - ``'notears'``
     - 
         - bias: bool
         - activation: str
         - rho_max: int
         - main_iter: int
         - h_tol: float
         - dims: typing.List
         - dtype: type
         - verbose: bool
   * - 
     - ``'dagma'``
     - 
         - T: int
         - mu_init: float
         - mu_factor: float
         - s: typing.Union[typing.List[float], float]
         - warm_iter: int
         - main_iter: int
         - dtype: type
         - dims: typing.List
         - bias: bool
         - verbose: bool
   * - 
     - ``'topo'``
     - 
         - no_large_search: int
         - size_small: int
         - size_large: int
         - topo: (list, type(None))
         - dims: typing.List
         - bias: bool
         - activation: str
         - dtype: type
         - verbose: bool


.. code-block:: python

    from dagrad import learn, generate_linear_data
    import numpy as np
    # Define the parameters of the data
    n, d, s0, graph_type, noise_type = 1000, 10, 10, 'ER', 'gauss'
    
    # Generate the data
    X, W_true, B_true = generate_linear_data(n, d, s0, graph_type, noise_type)
    
    # Define the model
    model = 'linear'

    # Define the method options
    method_options = {'verbose': True, 'h_tol': 1e-9, 'rho': 0.1'}
    
    # Learn the DAG using Notears, linear model, numpy, and adam optimizer
    W_est = learn(X, method='notears', model = model, optimizer = 'adam', method_options=method_options)

Optimizer Options
^^^^^^^^^^^^^^^^^^^^

The following options are available for different optimizers. Note that **opt_config** is the parameters for the optimizer and **opt_settings** are the hyperparameter for the whole optimization framework. Its specification can be done through the ``optimizer_options = {}``. 
The examples are given below.

.. note::
    Put all the options for ``opt_config`` and ``opt_settings`` together in the dictionary ``optimizer_options``. Please remember options for ``optimizer_options`` need be consistent with ``computed_lib`` and ``optimizer``.


.. list-table:: Allowed Optimizer Options
   :header-rows: 1
   :widths: 20 20 40

   * - ``computed_lib``
     - ``optimizer``
     - Configuration Options and Settings
   * - ``'numpy'``
     - ``'adam'``
     - **opt_config**:
         - 'lr': float
         - 'betas': typing.Tuple[float, float]
         - 'eps': float
       **opt_settings**:
         - 'check_iterate': int
         - 'tol': float
         - 'num_steps': int
   * - 
     - ``'lbfgs'``
     - **opt_config**:
         - 'disp': (int, None)
         - 'maxcor': int
         - 'ftol': float
         - 'gtol': float
         - 'eps': float
         - 'maxfun': int
         - 'maxiter': int
         - 'iprint': int
         - 'maxls': int
       **opt_settings**: 
         - {}
   * - 
     - ``'sklearn'``
     - **opt_config**: 
            - {}
       **opt_settings**: 
            - {}
   * - ``'torch'``
     - ``'adam'``
     - **opt_config**:
         - 'lr': float
         - 'betas': typing.Tuple[float, float]
         - 'eps': float
       **opt_settings**:
         - 'tol': float
         - 'num_steps': int
         - 'check_iterate': int
         - 'lr_decay': bool
   * - 
     - ``'lbfgs'``
     - **opt_config**:
         - 'lr': float
         - 'max_iter': int
         - 'max_eval': int or None
         - 'tolerance_grad': float
         - 'tolerance_change': float
         - 'history_size': int
         - 'line_search_fn': str or None
       **opt_settings**:
         - 'num_steps': int
         - 'tol': float
         - 'check_iterate': int
         - 'lr_decay': bool


.. code-block:: python

    from dagrad import learn, generate_linear_data
    import numpy as np
    # Define the parameters of the data
    n, d, s0, graph_type, noise_type = 1000, 10, 10, 'ER', 'gauss'
    
    # Generate the data
    X, W_true, B_true = generate_linear_data(n, d, s0, graph_type, noise_type)
    
    # Define the model
    model = 'linear'

    # Define the method options
    optimizer_options = {'lr':0.01, betas:(0.99, 0.999), 'eps':1e-8, 'check_iterate': 100, 'tol': 1e-6, 'num_steps': 10000}
    
    # Learn the DAG using Notears, linear model, numpy, and adam optimizer
    W_est = learn(X, method='notears', model = model, computed_lib = 'numpy', optimizer = 'adam', optimizer_options=optimizer_options)


.. [1] Deng, Chang, Kevin Bello, Pradeep Kumar Ravikumar, and Bryon Aragam. "Likelihood-based differentiable structure learning" Advances in Neural Information Processing Systems 37 (2024)