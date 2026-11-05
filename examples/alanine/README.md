# Alanine dipeptide in molecular environments

A prior trained on the backbone torsions $(\phi,\psi)$ of isolated alanine dipeptide is coupled to OpenMM dynamics of all atoms. Sodium coordination changes the carbonyl O–O distance; two monomer priors form a hydrogen-bonded dimer with anti-parallel and parallel topologies. The figure compares O–O distances and dimer torsion differences with MD.

The context is Amber99SB-ILDN in vacuum, **without the backbone terms** the prior replaces to relieve the double-counting.

## How it works

The state $s$ holds all atom coordinates. Each peptide's torsions $\Phi_i(s)=(\phi_i,\psi_i)$ are coupled to a clean sample $x_i$ of the prior through the wrapped Gaussian kernel $q_t$:

$$
\pi_t(s,\{x_i\})\propto e^{-U_{\mathrm{ctx}}(s)/k_BT}\prod_i p(x_i)\,q_t\big(\Phi_i(s)\mid x_i\big).
$$

A sweep denoises each peptide's torsions with the pretrained model, $x_i\sim p(x_i)\,q_t(\Phi_i(s)\mid x_i)$, then runs MD under

$$
U_{\mathrm{eff}}(s)=U_{\mathrm{ctx}}(s)-k_BT\sum_i\log q_t\big(\Phi_i(s)\mid x_i\big),
$$

that is, the edited force field plus torsional restraints toward the clean samples. As $t\to0$ the target becomes $e^{-U_{\mathrm{ctx}}/k_BT}\prod_i p(\Phi_i(s))$. The dimer adds replica exchange over $t$, with the swap rule of the [$\phi^4$ example](../phi4/README.md).

The prior has little coverage of $\phi>0$ conformations; dimer state occupancies require longer sampling.

## Run

From the repository root:

```bash
python -m pip install ".[md,figures]"
python examples/alanine/example_alanine.py --config examples/alanine/settings.yaml
```

Allow about 10–15 minutes. Settings are in [settings.yaml](settings.yaml); outputs go to `runs/alanine_<timestamp>/`.

## Train and sample

[train_and_sample.ipynb](train_and_sample.ipynb) trains the torsion prior from the vacuum MD torsions and runs this example with it. From the repository root:

```bash
python -m pip install ".[md,figures]" jupyter
jupyter nbconvert --to notebook --execute --inplace examples/alanine/train_and_sample.ipynb
```

Allow about 5 minutes on CPU; outputs go to `runs/alanine_notebook_<timestamp>/`.

See the [root README](../../README.md) for the figure and [configs/alanine.yaml](../../configs/alanine.yaml) for the paper settings.
