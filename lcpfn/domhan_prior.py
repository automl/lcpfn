from functools import partial
import torch
import numpy as np
from lcpfn.curves import (
    pow3,
    ilog2,
    janoschek,
    log_power,
    prior_ilog2,
    uniform_prior_pow3,
    weibull,
    mmf,
    vap,
    loglog_linear,
    exp4,
    pow4,
    dr_hill_zero_background,
)
from lcpfn.curves import (
    prior_pow3,
    prior_janoschek,
    prior_log_power,
    prior_weibull,
    prior_mmf,
    prior_vap,
    prior_loglog_linear,
    prior_exp4,
    prior_pow4,
    prior_dr_hill_zero_background,
)
from lcpfn.curves import (
    uniform_prior_pow3,
    uniform_prior_ilog2,
    uniform_prior_janoschek,
)


def prior_weights(
    rng,
    components=[
        "pow3",
        "ilog2",
        "janoschek",
        "log_power",
        "weibull",
        "mmf",
        "vap",
        "loglog_linear",
        "exp4",
        "pow4",
        "dr_hill_zero_background",
    ],
):
    K = len(components)
    weights = rng.uniform(0.0, 1, size=(K,))
    return {f: weights[i] for i, f in enumerate(components)}


def sample_from_prior(rng, seq_len=100):
    return sample_prior_comb(
        rng=rng,
        seq_len=seq_len,
        components=["pow3", "ilog2", "janoschek"],
        distribution="peaked",
    )


def sample_prior_comb(
    rng,
    components,
    distribution,
    var_lnloc=-4,
    var_lnscale=1,
    range_constraint=True,
    seq_len=100,
):
    f_components = {
        "pow3": pow3,
        "ilog2": ilog2,
        "janoschek": janoschek,
        "log_power": log_power,
        "weibull": weibull,
        "mmf": mmf,
        "vap": vap,
        "loglog_linear": loglog_linear,
        "exp4": exp4,
        "pow4": pow4,
        "dr_hill_zero_background": dr_hill_zero_background,
    }

    if distribution == "peaked":
        f_priors = {
            "pow3": prior_pow3,
            "ilog2": prior_ilog2,
            "janoschek": prior_janoschek,
            "log_power": prior_log_power,
            "weibull": prior_weibull,
            "mmf": prior_mmf,
            "vap": prior_vap,
            "loglog_linear": prior_loglog_linear,
            "exp4": prior_exp4,
            "pow4": prior_pow4,
            "dr_hill_zero_background": prior_dr_hill_zero_background,
        }
    elif distribution == "uniform":
        f_priors = {
            "pow3": uniform_prior_pow3,
            "ilog2": uniform_prior_ilog2,
            "janoschek": uniform_prior_janoschek,
        }
    else:
        raise NotImplemented()

    x = np.arange(1, seq_len + 1)

    while True:
        # sample the noiseless curve
        weights = prior_weights(rng, components=components)
        y = np.zeros(x.shape, dtype="float")
        kwargs = 0
        for f, w in weights.items():
            kwargs = f_priors[f](rng)
            # print(f_components[f](x, **kwargs))
            y += w * f_components[f](x, **kwargs)
        # add noise (can exceed [0,1], but afaik no way to implement this prior in Tobias work)
        # Note: This is the correct definition, but it differs from the noise prior definition in the paper
        std = np.exp(
            rng.normal(var_lnloc, var_lnscale)
        )  

        # reject any curves that are non-increasing, exceed the [0,1] range
        if (
            y[-1] <= y[0]
            or (range_constraint and (np.any(y < 0) or np.any(y > 1)))
            or np.isnan(y).any()
        ):
            continue
        else:
            break

    def curve():  # generates a sample from the same model, but with independent noise
        y_noisy = y + rng.normal(np.zeros_like(y), std)
        return y, y_noisy

    return curve


def generate_prior_dataset(n, prior=sample_prior_comb, seed=42):
    """
    Returns a fixed sample from the prior (with fixed seq_len) as an n x seq_len np.ndarray
    """
    rng = np.random.RandomState(seed)
    prior_data = np.stack([prior(rng)()[1] for _ in range(n)])
    return prior_data


def create_get_batch_func(prior):
    return partial(get_batch_domhan, prior=prior)


# function producing batches for PFN training
def get_batch_domhan(
    batch_size,
    seq_len,
    num_features,
    prior,
    device="cpu",
    noisy_target=True,
    **_,
):
    assert num_features == 1

    x = np.arange(1, seq_len + 1)
    y_target = np.empty((batch_size, seq_len), dtype=float)
    y_noisy = np.empty((batch_size, seq_len), dtype=float)

    for i in range(batch_size):
        curve_func = prior(np.random, seq_len=seq_len)  # uses numpy rng
        if noisy_target:
            _, y_noisy[i] = curve_func()
            y_target[i] = y_noisy[i]
        else:
            y_target[i], y_noisy[i] = curve_func()

    # turn numpy arrays into correctly shaped torch tensors & move them to device
    x = (
        torch.arange(1, seq_len + 1)
        .repeat((num_features, batch_size, 1))
        .transpose(2, 0)
        .to(device)
    )
    y_target = torch.from_numpy(y_target).transpose(1, 0).to(device)
    y_noisy = torch.from_numpy(y_noisy).transpose(1, 0).to(device)

    # changes
    x = x.float()
    y_target = y_target.float()
    y_noisy = y_noisy.float()

    return x, y_noisy, y_target
