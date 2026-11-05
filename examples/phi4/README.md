# Composing a $\phi^4$ lattice

An isolated-site prior, $p_0(\phi)\propto e^{-(\phi^2-1)^2}$, is composed through periodic nearest-neighbor interactions:

$$
H(\phi)=\sum_i(\phi_i^2-1)^2+J\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2-h\sum_i\phi_i,
\qquad p(\phi)\propto e^{-H(\phi)}.
$$

The example compares fields and absolute magnetization $|m|$ with checkerboard Metropolis at three couplings across the transition.

## How it works

Every site carries a copy of $p_0$, coupled to an auxiliary field $\psi$ by $q_t(\psi_i\mid\phi_i)=\mathcal{N}(\psi_i;\alpha_t\phi_i,\sigma_t^2)$. The quadratic coupling $J\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2$ is diagonal in Fourier space (lattice Laplacian eigenvalues $\lambda_k$) and is deconvolved mode by mode:

$$
q_{\mathrm{ctx}}(\psi;t)\propto\exp\Big[-\tfrac12\sum_k q_k(t)\,|\hat\psi_k|^2+\frac{h}{\alpha_t}\sum_i\psi_i\Big],\qquad q_k(t)=\frac{2J\lambda_k}{\alpha_t^2-2J\lambda_k\sigma_t^2}.
$$

A sweep denoises every site independently, $\phi_i\sim p_0(\phi_i)\,q_t(\psi_i\mid\phi_i)$, then draws $\psi$ exactly, mode by mode, from $q_{\mathrm{ctx}}(\psi;t)\prod_i q_t(\psi_i\mid\phi_i)$.

**Replica exchange.** Replicas at $t_1<\dots<t_R$ share the context fixed at $t_1$. Neighbours swap with probability

$$
\min\Big(1,\frac{q_{t_{r+1}}(\psi^{(r)}\mid\phi^{(r)})\,q_{t_r}(\psi^{(r+1)}\mid\phi^{(r+1)})}{q_{t_r}(\psi^{(r)}\mid\phi^{(r)})\,q_{t_{r+1}}(\psi^{(r+1)}\mid\phi^{(r+1)})}\Big),
$$

where the priors and the shared context cancel. Noisier replicas see a weaker effective coupling and cross the transition quickly.

## Run

With CUDA-enabled PyTorch, run from the repository root:

```bash
python -m pip install ".[figures]"
python examples/phi4/example_phi4.py --config examples/phi4/settings.yaml
```

Allow about 15 minutes. Settings are in [settings.yaml](settings.yaml); outputs go to `runs/phi4_<timestamp>/`.

## Train and sample

[train_and_sample.ipynb](train_and_sample.ipynb) trains the single-site prior from exact samples of $p_0$ and runs this example with it. From the repository root:

```bash
python -m pip install ".[figures]" jupyter
jupyter nbconvert --to notebook --execute --inplace examples/phi4/train_and_sample.ipynb
```

Allow about 20 minutes; outputs go to `runs/phi4_notebook_<timestamp>/`.

See the [root README](../../README.md) for the figure and [configs/phi4.yaml](../../configs/phi4.yaml) for the paper settings.
