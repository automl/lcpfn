import math

from torch import nn

from lcpfn import bar_distribution, encoders, train
from lcpfn import utils

from lcpfn.priors import utils as putils


def train_lcpfn(
    get_batch_func,
    seq_len: int = 100,
    emsize: int = 512,
    nlayers: int = 12,
    num_borders: int = 1000,
    lr: float = 0.0001,
    batch_size: int = 100,
    epochs: int = 1000,
):
    """
    Train a LCPFN model using the specified hyperparameters.

    Args:
        get_batch_func (callable): A function that returns a batch of learning curves.
        seq_len (int, optional): The length of the input sequence. Defaults to 100.
        emsize (int, optional): The size of the embedding layer. Defaults to 512.
        nlayers (int, optional): The number of layers in the model. Defaults to 12.
        num_borders_choices (int, optional): The number of borders to use. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.0001.
        batch_size (int, optional): The batch size for training. Defaults to 100.
        epochs (int, optional): The number of epochs to train for. Defaults to 1000.

    Returns:
        torch.module: The trained model.
    """

    hps = {}

    # PFN training hyperparameters
    dataloader = putils.get_batch_to_dataloader(get_batch_func)  # type: ignore

    num_features = 1

    ys = get_batch_func(
        10_000,
        seq_len,
        num_features,
        hyperparameters=hps,
        single_eval_pos=seq_len,
    )

    bucket_limits = bar_distribution.get_bucket_limits(num_borders, ys=ys[2])

    # Discretization of the predictive distributions
    criterions = {
        num_features: {
            num_borders: bar_distribution.FullSupportBarDistribution(bucket_limits)
        }
    }

    config = dict(
        nlayers=nlayers,
        priordataloader_class=dataloader,
        criterion=criterions[num_features][num_borders],
        encoder_generator=lambda in_dim, out_dim: nn.Sequential(
            encoders.Normalize(0.0, 101.0),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            encoders.Linear(in_dim, out_dim),
        ),
        emsize=emsize,
        nhead=(emsize // 128),
        warmup_epochs=(epochs // 4),
        y_encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        batch_size=batch_size,
        scheduler=utils.get_cosine_schedule_with_warmup,
        extra_prior_kwargs_dict={
            # "num_workers": 10,
            "num_features": num_features,
            "hyperparameters": {
                **hps,
            },
        },
        epochs=epochs,
        lr=lr,
        bptt=seq_len,
        single_eval_pos_gen=utils.get_uniform_single_eval_pos_sampler(
            seq_len, min_len=1
        ),
        aggregate_k_gradients=1,
        nhid=(emsize * 2),
        steps_per_epoch=100,
        train_mixed_precision=False,
    )

    return train.train(**config)
