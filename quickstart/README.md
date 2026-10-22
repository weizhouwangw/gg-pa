# CPU quickstarts

```bash
bash quickstart.sh
```

Requires Python 3.10+ with PyTorch, NumPy, SciPy, Matplotlib and PyYAML; no installation. Both examples take about five minutes on CPU and write `manifest.json`, `results/` and `figures/` to `runs/quickstart_<timestamp>/`.

## How it works

A pretrained prior $p(x)$ is coupled to the system state $s$ through its own forward kernel $q_t(s\mid x)$; the physics enters through a context factor $q_{\mathrm{ctx}}$:

$$
\pi_t(s,x)\propto q_{\mathrm{ctx}}(s,t)\,p(x)\,q_t(s\mid x).
$$

Gibbs sampling alternates $x\sim p(x)\,q_t(s\mid x)$, a denoising step done by the pretrained model, and $s\sim q_{\mathrm{ctx}}(s,t)\,q_t(s\mid x)$. Integrating out $s$ convolves the context with the kernel; for a Gaussian context this convolution can be made to equal the physical factor exactly.

## Linear inverse problem

Prior: $p(x)=\frac{1}{25}\sum_{\mu\in G^2}\mathcal{N}(x;\mu,0.8^2I)$ with $G=\{-5,-2.5,0,2.5,5\}$. Observation: $y=Hx+\eta$, observed at $y=0$, with $H=[1,0]$ and $\eta\sim\mathcal{N}(0,\Sigma_\eta)$, $\Sigma_\eta=2.5^2$.

Split Gibbs introduces $s$ with $q(s\mid x)=\mathcal{N}(s;x,\rho^2I)$ and the context $\mathcal{N}(y;Hs,\Sigma_{\mathrm{split}})$. The convolution is

$$
\int\mathcal{N}(y;Hs,\Sigma_{\mathrm{split}})\,\mathcal{N}(s;x,\rho^2I)\,ds=\mathcal{N}\big(y;Hx,\Sigma_{\mathrm{split}}+\rho^2HH^\top\big),
$$

so GG-PA sets $\Sigma_{\mathrm{split}}=\Sigma_\eta-\rho^2HH^\top$ and samples the exact posterior. Uncorrected split Gibbs uses $\Sigma_{\mathrm{split}}=\Sigma_\eta$ and samples an inflated likelihood.

## Double well

The prior is trained on $U_0(x)=8(x^2-1)^2$ and coupled to an environment:

$$
U(x,x_{\mathrm{env}})=8(x^2-1)^2+2(x-x_{\mathrm{env}})^2+\tfrac12(x_{\mathrm{env}}-1)^2.
$$

The reference marginal is $\pi(x)\propto\exp[-8(x^2-1)^2-0.4(x-1)^2]$. With $q_t(s\mid x)=\mathcal{N}(s;\alpha_tx,\sigma_t^2)$, the coupling $2(x-x_{\mathrm{env}})^2$ (precision $k_c=4$) is deconvolved into the context

$$
U_{\mathrm{ctx}}(s,x_{\mathrm{env}};t)=\tfrac12\kappa_t\,(s-\alpha_tx_{\mathrm{env}})^2+\tfrac12(x_{\mathrm{env}}-1)^2,\qquad \kappa_t=\frac{k_c}{\alpha_t^2-k_c\sigma_t^2},
$$

which is exact while $\alpha_t^2>k_c\sigma_t^2$. GG-PA-RE adds replica exchange over diffusion time.

Settings are in [run.py](run.py).
