
# Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

This repository offers an implementation of [LC-PFN](https://openreview.net/pdf?id=xgTV6rmH6n), a method designed for efficient Bayesian learning curve extrapolation.

**LC-PFN in action on [Google colab](https://colab.research.google.com/drive/1JA2t91xgqZVfjZya41oW5vVQktv_YXhE?usp=sharing) and [HuggingFace](https://huggingface.co/spaces/herilalaina/lcpfn)**

Installation using pip:

```bash
pip install -U lcpfn
```

### Usage

Try out the `notebooks` (require ``matplotlib``) for training and inference examples.

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


