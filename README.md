JAX kernel of interacting particle guidance (IPG) for reward-tilted generative priors (arXiv:2609.37227).

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

Not the 256-d GMM, not 500 time steps, no image inpainting, no protein structures, no SMC/FKC/DriftLite baselines, and no neural corrector. The smoke is a 2-d mixture with 32 particles.
