==========
Main API
==========

The following are the main function of the **dagrad** library. This is main function that calls the :ref:`working_function` of the library.
User can explore different options to customize the learning process, by specifiying the method, the model, the loss function, the acyclicity constraint, the regularizer and the optimizer.

.. autofunction:: dagrad.learn

.. note:: :code:`method_options` and :code:`optimizer_options` are dictionaries that can be used to specify the options for the method and the optimizer. 
            If the method or the optimizer is not specified, the default value are used. If you want to learn how to customize the method or the optimizer, please refer to :doc:`options`.

The following functions are wrappers for the NOTEARS [1]_ [2]_ [5]_, DAGMA [3]_, and TOPO [4]_. When the following function are used, they will call the **original** implementation of the methods except for the **TOPO** method. The **TOPO** method implemented in the **dagrad** library works better than original implementation, so it is recommended to use the **TOPO** method from the **dagrad** library.

.. autofunction:: dagrad.notears
.. autofunction:: dagrad.dagma
.. autofunction:: dagrad.topo

.. [1] Zheng, Xun, Bryon Aragam, Pradeep K. Ravikumar, and Eric P. Xing. "Dags with no tears: Continuous optimization for structure learning." Advances in neural information processing systems 31 (2018).
.. [2] Zheng, Xun, Chen Dan, Bryon Aragam, Pradeep Ravikumar, and Eric Xing. "Learning sparse nonparametric dags." In International Conference on Artificial Intelligence and Statistics, pp. 3414-3425. Pmlr, 2020.
.. [3] Bello, Kevin, Bryon Aragam, and Pradeep Ravikumar. "Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization." Advances in Neural Information Processing Systems 35 (2022): 8226-8239.
.. [4] Deng, Chang, Kevin Bello, Bryon Aragam, and Pradeep Kumar Ravikumar. "Optimizing NOTEARS objectives via topological swaps." In International Conference on Machine Learning, pp. 7563-7595. PMLR, 2023.
.. [5] Deng, Chang, Kevin Bello, Pradeep Kumar Ravikumar, and Bryon Aragam. "Likelihood-based differentiable structure learning" Advances in Neural Information Processing Systems 37 (2024)
