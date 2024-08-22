import os
import math
import argparse
import random
import datetime

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


# copied from huggingface
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# copied from huggingface
def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def get_weighted_single_eval_pos_sampler(max_len):
    """
    This gives a sampler that can be used for `single_eval_pos` which yields good performance for all positions p,
    where p <= `max_len`. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(
        range(max_len), [1 / (max_len - i) for i in range(max_len)]
    )[0]


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def set_locals_in_self(locals):
    """
    Call this function like `set_locals_in_self(locals())` to set all local variables as object variables.
    Especially useful right at the beginning of `__init__`.
    :param locals: `locals()`
    """
    self = locals["self"]
    for var_name, val in locals.items():
        if var_name != "self":
            setattr(self, var_name, val)


default_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


# Copied from StackOverflow, but we do an eval on the values additionally
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = eval(v)
            except NameError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        print("dict values: {}".format(my_dict))


def get_nan_value(v, set_value_to_nan=0.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def to_ranking(data):
    x = data >= data.unsqueeze(-3)
    x = x.sum(0)
    return x


# TODO: Is there a better way to do this?
#   1. Cmparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def nan_handling_missing_for_unknown_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float("nan"), set_value_to_nan)


def nan_handling_missing_for_no_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float("-inf"), set_value_to_nan)


def nan_handling_missing_for_a_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float("inf"), set_value_to_nan)


def torch_nanmean(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    return value / num


def torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis), x.shape[axis], dim=axis
    )
    return torch.sqrt(
        torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1)
    )


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], axis=0)
        std = torch_nanstd(data[:normalize_positions], axis=0) + 0.000001
    else:
        mean = torch_nanmean(data, axis=0)
        std = torch_nanstd(data, axis=0) + 0.000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def remove_outliers(X, n_sigma=4):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    data = X
    data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data_clean = X[:].clone()
    data_clean[torch.logical_or(data > upper, data < lower)] = np.nan
    data_mean, data_std = (
        torch_nanmean(data_clean, axis=0),
        torch_nanstd(data_clean, axis=0),
    )
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


def bool_mask_to_att_mask(mask):
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device):
    print("init dist")
    if "LOCAL_RANK" in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )
        return True, rank, f"cuda:{rank}"
    elif "SLURM_PROCID" in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        assert device != "cpu:0"
        rank = int(os.environ["SLURM_PROCID"])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        print("distributed submitit launch and my rank is", rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )

        return True, rank, f"cuda:{rank}"
    else:
        print("Not using distributed")
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, device


def check_compatibility(dl):
    if hasattr(dl, "num_outputs"):
        print(
            "`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on."
        )
        assert dl.num_outputs != 1, (
            "We assume num_outputs to be 1. Instead of the num_ouputs change your loss."
            "We specify the number of classes in the CE loss."
        )


def pfn_normalize(
    lb=torch.tensor(float("-inf")),
    ub=torch.tensor(float("inf")),
    soft_lb=0.0,
    soft_ub=1.0,
    minimize=False,
):
    """
    LC-PFN curve prior assumes curves to be normalized within the range [0,1] and to be maximized.
    This function allows to normalize and denormalize data to fit this assumption.

    Parameters:
        lb (torch.Tensor): Lower bound of the data.
        ub (torch.Tensor): Upper bound of the data.
        soft_lb (float): Soft lower bound for normalization. Default is 0.0.
        soft_ub (float): Soft upper bound for normalization. Default is 1.0.
        minimize (bool): If True, the original curve is a minization. Default is False.

    Returns: Two functions for normalizing and denormalizing the data.
    """
    assert lb <= soft_lb and soft_lb < soft_ub and soft_ub <= ub
    # step 1: linearly transform [soft_lb,soft_ub] [-1,1] (where the sigmoid behaves approx linearly)
    #    2.0/(soft_ub - soft_lb)*(x - soft_lb) - 1.0
    # step 2: apply a vertically scaled/shifted the sigmoid such that [lb,ub] --> [0,1]

    def cinv(x):
        return 1 - x if minimize else x

    def lin_soft(x):
        return 2 / (soft_ub - soft_lb) * (x - soft_lb) - 1

    def lin_soft_inv(y):
        return (y + 1) / 2 * (soft_ub - soft_lb) + soft_lb

    try:
        if torch.exp(-lin_soft(lb)) > 1e300:
            raise RuntimeError
        # otherwise overflow causes issues, treat these cases as if the lower bound was -infinite
        # print(f"WARNING: {lb} --> NINF to avoid overflows ({np.exp(-lin_soft(lb))})")
    except RuntimeError:
        lb = torch.tensor(float("-inf"))
    if torch.isinf(lb) and torch.isinf(ub):
        return lambda x: cinv(
            1 / (1 + torch.exp(-lin_soft(x)))
        ), lambda y: lin_soft_inv(torch.log(cinv(y) / (1 - cinv(y))))
    elif torch.isinf(lb):
        a = 1 + torch.exp(-lin_soft(ub))
        return lambda x: cinv(
            a / (1 + torch.exp(-lin_soft(x)))
        ), lambda y: lin_soft_inv(torch.log((cinv(y) / a) / (1 - (cinv(y) / a))))
    elif torch.isinf(ub):
        a = 1 / (1 - 1 / (1 + torch.exp(-lin_soft(lb))))
        b = 1 - a
        return lambda x: cinv(
            a / (1 + torch.exp(-lin_soft(x))) + b
        ), lambda y: lin_soft_inv(
            torch.log(((cinv(y) - b) / a) / (1 - ((cinv(y) - b) / a)))
        )
    else:
        a = (
            1
            + torch.exp(-lin_soft(ub))
            + torch.exp(-lin_soft(lb))
            + torch.exp(-lin_soft(ub) - lin_soft(lb))
        ) / (torch.exp(-lin_soft(lb)) - torch.exp(-lin_soft(ub)))
        b = -a / (1 + torch.exp(-lin_soft(lb)))
        return lambda x: cinv(
            a / (1 + torch.exp(-lin_soft(x))) + b
        ), lambda y: lin_soft_inv(
            torch.log(((cinv(y) - b) / a) / (1 - ((cinv(y) - b) / a)))
        )


def get_default_normalizer():
    default_normalizer_kwargs = {
        "lb": torch.tensor(0.0),
        "ub": torch.tensor(1.0),
        "soft_lb": 0.0,
        "soft_ub": 1.0,
        "minimize": False,
    }
    return pfn_normalize(**default_normalizer_kwargs)


def identity_normalizer():
    return lambda x: x, lambda x: x
