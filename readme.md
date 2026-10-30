
# Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

This repository offers an implementation of [LC-PFN](https://openreview.net/pdf?id=xgTV6rmH6n), a method designed for efficient Bayesian learning curve extrapolation.

**LC-PFN in action on [Google colab](https://colab.research.google.com/drive/1JA2t91xgqZVfjZya41oW5vVQktv_YXhE?usp=sharing) and [HuggingFace](https://huggingface.co/spaces/herilalaina/lcpfn)**

Installation using pip:

```bash
pip install -U lcpfn
```

> **Update**: there is an inconsistency between the code and the paper regarding the definition of the noise prior. The correct definition is the one used in the code, where $\log(\sigma)$ is defined as $\mathcal{N}(-4, 1)$.

### Usage

Try out the `notebooks` (require ``matplotlib``) for training and inference examples.

```python
import torch
from lcpfn import LCPFN

model = LCPFN()  # default: EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000, downloaded on first use
x_train = torch.arange(1, 11).unsqueeze(-1)       # epochs observed so far
y_train = torch.linspace(0.1, 0.5, 10).unsqueeze(-1)  # increasing curve, values in [0, 1]
x_test = torch.arange(11, 21).unsqueeze(-1)       # epochs to extrapolate to
mean = model.predict_mean(x_train, y_train, x_test)
q10, q50, q90 = model.predict_quantiles(x_train, y_train, x_test, qs=[0.1, 0.5, 0.9]).T
```

#### Model variants

All 54 LC-PFN variants evaluated in the paper (Table 2) are released and can be selected by name,
`EPOCH{10,100,1000}_EMSIZE{128,256,512}_NLAYERS{3,6,12}_NBUCKETS{1000,10000}`
(training epochs x embedding size x transformer layers x output buckets of the prediction head):

```python
import lcpfn
lcpfn.list_models()                                   # the 54 names
model = lcpfn.LCPFN("EPOCH1000_EMSIZE256_NLAYERS6_NBUCKETS1000")  # smaller/faster model
model = lcpfn.LCPFN("/path/to/my_checkpoint.pt")      # or a checkpoint trained with train_lcpfn
```

Weights are downloaded on first use from https://ml.informatik.uni-freiburg.de/research-artifacts/lcpfn/
into `lcpfn/trained_models/`. Names without the `EPOCH` part (e.g. the default `EMSIZE512_NLAYERS12_NBUCKETS1000`)
refer to the 1000-epoch models. The paper's main model is `EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000`.

> **Note (fixed in 0.1.4):** from Aug 2024 to Oct 2026 the file served under the 6-layer name
> (`EMSIZE512_NLAYERS6_NBUCKETS1000`) was a copy of the 12-layer weights. lcpfn>=0.1.4 detects such a
> stale download and fetches the correct 6-layer model; with older versions, delete
> `lcpfn/trained_models/pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000.pt*` to force a re-download.

**NOTE:**  Our model supports only increasing curves with values in $[0,1]$. If needed, please consider normalizing your curves to meet these constraints. See an example in ``notebooks/curve_normalization.ipynb``.


### Reference

```
@inproceedings{
adriaensens2023lcpfn,
title={Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks},
author={Adriaensen, Steven and Rakotoarison, Herilalaina and Müller, Samuel and Hutter, Frank},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=xgTV6rmH6n}
}
```


