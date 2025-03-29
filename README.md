# dagrad: Gradient-based structure learning for causal DAGs

`dagrad` is a Python package that provides an extensible, modular platform for developing and experimenting with differentiable (gradient-based) structure learning methods. 

It builds upon the [NOTEARS][notears] framework and also functions as an updated repository of state-of-the-art implementations for various methods.

`dagrad` provides the following key features:

- A **universal** framework for implementing general score-based methods that are end-to-end differentiable 
- A **modular** implementation that makes it easy to swap out different score functions, acyclicity constraints, regularizers, and optimizers
- An **extensible** approach that allows users to implement their own objectives and constraints using PyTorch
- **Speed and scalability** enabled through GPU acceleration

## Introduction 

A directed acyclic graphical model (also known as a Bayesian network) with `d` nodes represents a distribution over a random vector of size `d`. The focus of this library is on Bayesian Network Structure Learning (BNSL): Given samples $\mathbf{X}$ from a distribution, how can we estimate the underlying graph `G`?

The problem can be formulated as the following differentiable continuous optimization:

```math
\begin{array}{cl}
\min _{W \in \mathbb{R}^{d \times d}} & Q(W;\mathbf{X}) \\
\text { subject to } & h(W) = 0,
\end{array}
```
This formulation is versatile enough to encompass both linear and nonlinear models with any smooth objective (e.g. log-likelihood, least-squares, cross-entropy, etc.).

`dagrad` provides a unified framework that allows users to employ either predefined or customized loss functions $Q(W;\mathbf{X})$, acyclicity constraints $h(W)$, and choose between first-order or second-order optimizers:
```math
\text{loss~function} + \text{acyclicity~constraint} + \text{optimizer} \implies \text{structure~learning}
```

GPU acceleration is also supported.

## Installation
To install `dagrad`:

```bash
$ git clone https://github.com/Duntrain/dagrad.git
$ cd dagrad/
$ pip install -e .
$ cd tests
$ python test_fast.py
```

The above installs `dagrad` and runs the original NOTEARS method [[1]][notears] on a randomly generated 10-node Erdos-Renyi graph with 1000 samples. The output should look like the below:
```
{'fdr': 0.0, 'tpr': 1.0, 'fpr': 0.0, 'shd': 0, 'nnz': 10}
```

Want to try more examples? See an example in this [iPython notebook][examples].

## Quickstart
Here is a simple demo:
```bash
$ import dagrad 
$ n, d, s0, graph_type, sem_type = 1000, 10, 10, 'ER', 'gauss'
$ X, W, G = dagrad.generate_linear_data(n, d, s0, graph_type, sem_type)
$ W_est = dagrad.dagrad(X = X, model = 'linear', method = 'notears')
$ acc = dagrad.count_accuracy(G, W_est != 0)
$ print('Accuracy: ', acc)
```

## Features
Below is an overview of the functionalities provided by the package:


| __Method(`method`)__ | __Model(`model`)__ |__Loss(`loss_fn`)__ |__Regularizers(`reg`)__|__h(`h_fn`)__ |__Optimizer(`optimizer`)__  | __Computation Library(`compute_lib`)__ |__Device(`device`)__|
| --------   | --------  |----|---|-------------------|----| ----------| --------------| 
|`'notears'`[1]    | `'linear'`,<br>`'nonlinear'`   |`'l2'`, `'logll'`, `'logdetll_ev'`, `'logdetll_nv'`, `'user_loss'`|`'l1'`<br> `'l2'`<br> `'mcp'`<br> `'none'`<br>`'user_reg'` |`'h_exp_sq'`<br>`'h_poly_sq'`<br>`'h_poly_abs'`<br>`'user_h'` |Adam(`'adam'`),<br>LBFGS(`'lbfgs'`)        |  Numpy(`'numpy'`),<br>Torch(`'torch'`),  |  CPU(`'cpu'`)<br>CUDA(`'cuda'`)      | 
| `'dagma'`[2]      | `'linear'`,<br>`'nonlinear'`    |`'l2'`, <br>  `'logll'`, <br> `'logdetll_ev'`, <br> `'logdetll_nv'`, <br>`'user_loss'`|`'l1'`<br> `'l2'`<br> `'mcp'`<br> `'none'`, `'user_reg'`| `'h_logdet_sq'`<br>`'h_logdet_abs'`<br>`'user_h'` |Adam(`'adam'`)            |  Numpy(`'numpy'`)<br>Torch(`'torch'`)  |  CPU(`'cpu'`)<br>CUDA(`'cuda'`)      |
| `'topo'`[3]       |`'linear'`,<br>`'nonlinear'`   |`'l2'`,<br> `'logll'`,<br>`'user_loss'`| `'l1'`<br> `'l2'`<br> `'mcp'`<br> `'none'`<br> `'user_reg'` |`'h_exp_topo'`<br>`'h_logdet_topo'`<br>`'h_poly_topo'`<br>`'user_h'` |Adam(`'adam'`),<br> LBFGS(`'lbfgs'`)|  Numpy(`'numpy'`) for linear <br> Torch(`'torch'`) for nonlinear |  CPU(`'cpu'`)     | 


<!-- | __Method(`method`)__ | __Model(`model`)__ |__Loss(`loss_fn`)__ |__Regularizers(`reg`)__|__h(`h_fn`)__ |__Optimizer(`optimizer`)__ | __Computation Library(`compute_lib`)__|__Device(`device`)__|
| --------   | --------  |----|---|-------------------|----| ----------| --------------| 
| NOTEARS<br>(`'notears'`)[1]     | NonLinear(`'nonlinear'`)   |`'l2'`<br> `'logll'`<br> `'user_loss'`|`'l1'`<br> `'l2'`<br> `'mcp'`<br>`'none'`<br> `'user_reg'` |`'h_exp_sq'`<br>`'h_poly_sq'`<br>`'h_poly_abs'`<br>`'user_h'` |Adam(`'adam'`)<br>LBFGS(`lbfgs`)        |  Torch(`'torch'`)  |  CPU(`'cpu'`)<br>CUDA(`'cuda'`)      | 
| DAGMA<br>(`'dagma'`)[2]       | NonLinear(`'nonlinear'`)     |`'l2'`<br> `'logll'`<br> `'user_loss'`|`'l1'`<br> `'l2'`<br> `'mcp'`<br> `'none'`<br> `'user_reg'`| `'h_logdet_sq'`<br>`'h_logdet_abs'`<br>`'user_h'` |Adam(`'adam'`)<br>LBFGS(`lbfgs`)            |  Torch(`'torch'`)  |  CPU(`'cpu'`)<br>CUDA(`'cuda'`)      |
| TOPO<br>(`'topo'`)[3]      | NonLinear(`'nonlinear'`)     |`'l2'`<br> `'logll'`<br> `'user_loss'`| `'l1'`<br> `'l2'`<br> `'mcp'`<br>`'none'`<br> `'user_reg'` |`'h_exp_topo'`<br>`'h_logdet_topo'`<br>`'h_poly_topo'`<br>`'user_h'` |Adam(`'adam'`)<br>LBFGS(`lbfgs`)|  Torch(`'torch'`) |  CPU(`'cpu'`)     |  -->

- For the linear (`'linear'`) model, the loss function (`loss_fn`) can be configured as logistic loss (`'logistic'`) for all three methods.
- In the linear (`'linear'`) model, the default optimizer (`'optimizer'`) for TOPO (`'topo'`) is [scikit-learn](https://scikit-learn.org/stable/) (`'sklearn'`), a state-of-the-art package for solving linear model problems.
- In the linear (`'linear'`) model, NOTEARS (`'notears'`) and DAGMA (`'dagma'`) also support computation libraries (`compute_lib`) such as Torch (`'torch'`), and can perform computations on either CPU (`'cpu'`) or GPU (`'cuda'`).
## Requirements
- Python 3.7+
- `numpy`
- `scipy`
- `scikit-learn`
- `python-igraph`
- `tqdm`
- `dagma`
- `notears`: installed from github repo
- `torch`: Used for models with GPU acceleration



## References

[1] Zheng X, Aragam B, Ravikumar P, & Xing EP [DAGs with NO TEARS: Continuous optimization for structure learning][notears] (NeurIPS 2018, Spotlight).

[2] Zheng X, Dan C, Aragam B, Ravikumar P, & Xing EP [Learning sparse nonparametric DAGs][notearsmlp] (AISTATS 2020).

[3] Bello K, Aragam B, Ravikumar P [DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization][dagma] (NeurIPS 2022). 

[4] Deng C, Bello K, Aragam B, Ravikumar P [Optimizing NOTEARS Objectives via Topological Swaps][topo] (ICML 2023).

[5] Deng C, Bello K, Ravikumar P, Aragam B [Likelihood-based differentiable structure learning][logll] (NeurIPS 2024).

[notears]: https://arxiv.org/abs/1803.01422
[notearsmlp]: https://arxiv.org/abs/1909.13189
[dagma]: https://arxiv.org/abs/2209.08037
[topo]: https://arxiv.org/abs/2305.17277
[examples]: https://github.com/Duntrain/dagrad/blob/master/examples/examples.ipynb
[logll]: https://arxiv.org/pdf/2410.06163?
[notears_repo]: https://github.com/xunzheng/notears
