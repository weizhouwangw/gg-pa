# GGPA manual

GGPA composes conditional prior samplers with an explicit physical context. Start with the [quickstarts](../README.md#quick-start); this manual lists the interfaces for a new system.

## Target and one sweep

For system coordinates $s$, clean component variables $x_i$ and projections $\Phi_i$, the joint target is

$$
\pi(s,\{x_i\})\propto C(s)\prod_i p_i(x_i)q_{t_i}(\Phi_i(s)\mid x_i).
$$

A sweep draws each $x_i$ from its prior conditional given $\Phi_i(s)$, then updates $s$ under $C(s)\prod_i q_{t_i}(\Phi_i(s)\mid x_i)$, either exactly or with a transition that preserves this conditional. `state.prior_samples` holds the clean variables; `state.s` is the system state.

## Compose and inspect a sampler

```python
import torch
from ggpa import compose
from ggpa.contexts import gaussian
from ggpa.priors import mixture_posterior
from ggpa.priors.kernels import Gaussian, GeometricVE
from ggpa.projection import Select

kernel = Gaussian(GeometricVE(0.1, 1.0), (1,))
prior = mixture_posterior(torch.ones(1), torch.zeros(1, 1), torch.ones(1, 1, 1), kernel)
sampler = compose(
    [(prior, Select([[0]]))],
    gaussian(mean=[0.5], covariance=[[1.0]]),
    initial=torch.zeros(32, 1), t=0.2, seed=7,
)
state = sampler.step()
```

`compose(priors, context, *, initial=None, context_state=None, t=0.1, seed=0, dtype=torch.float64, device=None)` returns a `GibbsSampler`. Without `initial`, the context supplies the start through `initial_state()`.

| Snapshot field | Shape or meaning |
| --- | --- |
| `state.s` | System coordinates `(B, *state_shape)` |
| `state.prior_samples` | One clean tensor per attachment, `(B, M, *prior_shape)` |
| `state.context_state` | Auxiliary state such as MD velocities, else `None` |
| `state.step` | Completed sweeps |

Projections always include the copy axis `M`. Copies of one attachment share a diffusion time; use separate attachments for different times. Each sampler owns its random generator, and custom updates must use it.

## Priors, kernels and projections

| Component | Interface |
| --- | --- |
| `Prior` | `Prior(denoise, kernel)`; `denoise(y, t, rng)` draws `x` from the prior conditional |
| Conditional callable | `y`, `x`: `(B*M, *prior_shape)`; `t`: `(B*M,)` |
| Kernel | `shape`, `period`, `level(t)`, `log_q(y, x, level)` |
| Projection | Maps `(B, *state_shape)` to `(B, M, *prior_shape)`; declares a `period` matching the kernel |

`Gaussian` is $q_t(y\mid x)=\mathcal N(y;\alpha_t x,\sigma_t^2I)$ and `WrappedGaussian` its periodic form. `DiscreteVP` stores a checkpoint's cumulative-alpha table; `GeometricVE` interpolates between two noise scales.

| Projection in `ggpa.projection` | Convention |
| --- | --- |
| `Select(index)` | Indices `(M, d)` into flattened coordinates |
| `Linear(matrix, offset=None)` | Affine maps `(M, d, n)`, offsets `(M, d)` |
| `Torsions(quartets, ...)` | Atom quartets `(M, angles, 4)`; radians |
| `Cartesian(atoms, center=..., scale=1.0)` | Atoms `(M, N)`; optional centering (no rotation alignment); `scale` in nm per prior unit |

Load a bundled network prior:

```python
from ggpa.priors import ddpm, v_to_eps
from ggpa.priors.networks import load_net

net, kernel, metadata = load_net("checkpoints/doublewell_prior.pt", "cpu")
prior = ddpm(v_to_eps(net, kernel), kernel)
```

| Conditional sampler | Use |
| --- | --- |
| `ddpm` | Ancestral DDPM on the checkpoint's noise grid (1,000 levels for the lattice prior) |
| `torus_ancestral` | Ancestral sampler for wrapped-VE priors; its kernel has variance $\sigma(t)^2-\sigma_{\min}^2$, so $t>0$ |
| `mixture_posterior` | Exact Gaussian-mixture conditional |
| `grid_posterior` | Finite-grid conditional |

Any callable that satisfies the conditional contract can serve as a prior.

## Contexts

`Context` is a `Protocol`. Only `update(s, prior_samples, couplings, t, context_state, rng)` is required; it returns `(new_s, new_context_state)`.

| Factory in `ggpa.contexts` | Update |
| --- | --- |
| `quadratic(precision, linear=None)` | Exact Gaussian update for $C(s)\propto\exp(-s^TPs/2+b^Ts)$ |
| `gaussian(mean, covariance)` | The same in mean/covariance form |
| `deconvolved_quadratic(P, b, kernel=..., context_noise=...)` | Context whose convolution with the kernel at time `context_noise` equals $\exp(-x^TPx/2+b^Tx)$ |
| `from_energy(energy, step_size=..., steps=1)` | Metropolis-adjusted Langevin on the context plus coupling energy |
| `custom(update, ...)` | Wrap your own update function |

Lattice and double-well corrections are in `ggpa.systems.phi4` and `ggpa.systems.doublewell`.

### A custom transition

```python
from ggpa.contexts import coupling_energy, custom

def energy(s):
    return (0.15 * s**4 + 0.2 * s).flatten(1).sum(1)

def update(s, prior_samples, couplings, t, context_state, rng):
    def effective(z):
        return energy(z) + coupling_energy(z, prior_samples, couplings, t)

    proposed = s + 0.2 * torch.randn(s.shape, generator=rng, device=s.device, dtype=s.dtype)
    log_ratio = effective(s) - effective(proposed)
    accept = torch.log(torch.rand(len(s), generator=rng, device=s.device)) < log_ratio
    mask = accept.reshape(len(s), *([1] * (s.ndim - 1)))
    return torch.where(mask, proposed, s), context_state

context = custom(update, log_factor=lambda s: -energy(s))
```

Read `t` from each call; do not close over the initial value.

| Optional hook | Purpose |
| --- | --- |
| `validate(couplings, t, s)` | Compatibility checks at composition and state/time edits |
| `initial_state()` | Return `(s, context_state)` when `initial` is omitted |
| `log_factor(s)` | Context factor for exchange between different contexts |
| `exchange_key` | Equal keys mark the same factor |

### OpenMM

`md.context` takes an OpenMM System and coordinates in nm (temperature in K, time in ps, energy in kJ/mol) and owns its engines and velocities. See the [alanine example](../examples/alanine/README.md).

```python
from ggpa.contexts import openmm as md

context = md.context(
    system, initial=positions[None], temperature=300,
    dt=0.002, friction=1, steps=100, platform="Reference",
)
sampler = compose([(prior, projection)], context, t=0.1, seed=7)
```

The System must already exclude the interactions the prior replaces (`md.zero_nonbonded_within`, `md.zero_torsions_within`). The adapter adds the coupling $k_BT[-\log q_t]$ and checks it against the prior's kernel. `md.wrap` wraps a prepared Simulation instead.

## Sampling, exchange and restart

| Method | Operation |
| --- | --- |
| `step()` | Prior updates, then a context update |
| `update_priors()` / `update_context()` | One half of a sweep |
| `set_time(t)` | Scalar, or one entry per attachment |
| `set_state(s=..., prior_samples=..., context_state=...)` | Replace the supplied fields |
| `exchange(other)` | Attempt a full-state swap per batch row |
| `state_dict()` / `load_state_dict(record)` | Save or restore state, times and RNG |

Changing state or time is a warm start; recompose for a new shape, dtype, device or topology.

Replicas must share the same prior and projection objects. Contexts must be shared (the same object or equal `exchange_key`) or provide `log_factor`. Positions, clean variables and auxiliary state swap; times and RNG streams stay with their slots. MD replicas need separate engines.

```python
pairs = [(prior, projection)]
replicas = [compose(pairs, context, initial=initial, t=t, seed=7+i) for i, t in enumerate((0.1, 0.2))]
for replica in replicas:
    replica.step()
accepted = replicas[0].exchange(replicas[1])
```

## Checkpoint files

Bundled priors use `ggpa-prior/1` (`format`, `net`, `output`, `kernel`). Sampler snapshots use `ggpa-sampler/1`; they hold state, times and RNG, not model weights.

```python
torch.save(sampler.state_dict(), "sampler.pt")
sampler.load_state_dict(torch.load("sampler.pt", map_location="cpu", weights_only=True))
```

## Checking accuracy on your own system

1. Decide which interactions each prior replaces and remove them from the context.
2. Check units, periodicity and noise levels; the coupling must use the prior's kernel.
3. Test the conditional sampler against an exact or tractable case.
4. Vary step counts and integration settings.
5. Compare independent chains and starts; monitor exchange acceptance.

Replica slots are coupled; do not treat them as independent chains.

## Paper settings

[`configs/`](../configs/) holds the paper's lattice and alanine settings, in the format of the examples:

```bash
python examples/phi4/example_phi4.py --config configs/phi4.yaml
python examples/alanine/example_alanine.py --config configs/alanine.yaml
```
