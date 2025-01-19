==================
Getting started
==================

Installation
^^^^^^^^^^^^

Install via PyPI
------------------
.. code-block:: console

    $ pip install dagrad


Install from source
---------------------

For development version, you can install from source by cloning the repository and running the setup script:

.. code-block:: console

    $ git clone https://github.com/Duntrain/dagrad.git
    $ cd dagrad/
    $ pip install -e .
    $ cd tests
    $ python test_fast.py

Running examples
^^^^^^^^^^^^^^^^


Running the NOTEARS
-----------------------
.. code-block:: python

    # Import all the necessary modules
    from dagrad import learn # learn function is the main class for learning the structure of a DAG
    from dagrad import generate_linear_data, count_accuracy # utils is a module that contains some useful functions for generating data and measuring performance

    # Generate some linear data
    n, d, s0, graph_type, noise_type = 1000, 10, 10, 'ER', 'gauss' # Define the parameters of the data
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type) # Generate the data
    model = 'linear' # Define the model
    
    # Learn the structure of the DAG
    W_notears = learn(X, model = model, method = 'notears') # Learn the structure of the DAG using Notears
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, noise type: {noise_type}")
    
    # Measure the accuracy of the learned structure using Notears
    acc_notears = count_accuracy(B_true, W_notears != 0) 
    print('Accuracy of Notears:', acc_notears)


Running the DAGMA
------------------
.. code-block:: python

    # Import all the necessary modules
    from dagrad import learn # learn function is the main class for learning the structure of a DAG
    from dagrad import generate_linear_data, count_accuracy # utils is a module that contains some useful functions for generating data and measuring performance

    # Generate some linear data
    n, d, s0, graph_type, noise_type = 1000, 10, 10, 'ER', 'gauss' # Define the parameters of the data
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type) # Generate the data
    model = 'linear' # Define the model
    
    # Learn the structure of the DAG
    W_dagma = learn(X, model = model, method = 'dagma') # Learn the structure of the DAG using Dagma
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, noise type: {noise_type}")
    
    # Measure the accuracy of the learned structure using Dagma
    acc_dagma = count_accuracy(B_true, W_dagma != 0) 
    print('Accuracy of Dagma:', acc_dagma)


Running the TOPO
------------------
.. code-block:: python

    # Import all the necessary modules
    from dagrad import learn # learn function is the main class for learning the structure of a DAG
    from dagrad import generate_linear_data, count_accuracy # utils is a module that contains some useful functions for generating data and measuring performance

    # Generate some linear data
    n, d, s0, graph_type, noise_type = 1000, 10, 10, 'ER', 'gauss' # Define the parameters of the data
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type) # Generate the data
    model = 'linear' # Define the model
    
    # Learn the structure of the DAG
    W_topo = learn(X, model = model, method = 'topo') # Learn the structure of the DAG using Topo
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, noise type: {noise_type}")
    
    # Measure the accuracy of the learned structure using Topo
    acc_topo = count_accuracy(B_true, W_topo != 0) 
    print('Accuracy of Topo:', acc_topo)
