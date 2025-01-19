=======================
Customization
=======================


This section contains the customization options of the **dagrad** library. These options are used to customize the loss function (``'loss_fn'``), regularization (``'reg'``), and acyclicity functions (``'h'``). The following table shows the customization options and the location of the functions.

.. list-table:: Customization options
   :header-rows: 1

   * - Option
     - Description
     - Location
   * - ``'loss_fn'``
     - Loss function(linear)
     - :func:`dagrad.score.score.loss_fn.user_loss`
   * - ``'nl_loss_fn'``
     - Loss function(nonlinear)
     - :func:`dagrad.score.score.nl_loss_fn.user_nl_loss`
   * - ``'reg'``
     - Regularization
     - :func:`dagrad.score.score.reg_fn.user_reg`
   * - ``'h'``
     - Acyclicity function
     - :func:`dagrad.hfunction.h_functions.h_fn.user_h`

Go to the corresponding function and modify the function according to your needs. After you modified the loss function, regularization, and acyclicity function, you use them in the :func:`dagrad.learn` by setting the parameters ``loss_fn = 'user_loss'``, ``reg = 'user_reg'``, and ``h = 'user_h'``.

.. code-block:: python

    from dagrad import learn
    
    W_est = learn(X, loss_fn='user_loss', reg='user_reg', h='user_h')


Loss function(``'loss_fn'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linear model
------------

.. automethod:: dagrad.score.score.loss_fn.user_loss

Nonlinear model
---------------
.. automethod:: dagrad.score.score.nl_loss_fn.user_nl_loss



Regularization(``'reg'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: dagrad.score.score.reg_fn.user_reg


Acyclicity function (``'h'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: dagrad.hfunction.h_functions.h_fn.user_h

