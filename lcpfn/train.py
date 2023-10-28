import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext

import pickle
import torch
from torch import nn

from lcpfn import utils
from lcpfn.transformer import TransformerModel
from lcpfn.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_limits,
)
from lcpfn.utils import (
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    StoreDictKeyPair,
    get_weighted_single_eval_pos_sampler,
    get_uniform_single_eval_pos_sampler,
)
from lcpfn import priors
from lcpfn import encoders
from lcpfn import positional_encodings
from lcpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler


class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = lambda num_classes: nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(num_classes)
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution


def train(
    priordataloader_class,
    criterion,
    encoder_generator,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    dropout=0.2,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    bptt=10,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    input_normalization=False,
    y_encoder_generator=None,
    pos_encoder_generator=None,
    decoder=None,
    extra_prior_kwargs_dict={},
    scheduler=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    validation_period=10,
    single_eval_pos_gen=None,
    bptt_extra_samples=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=True,
    style_encoder_generator=None,
    epoch_callback=None,
    initializer=None,
    initialize_with_model=None,
    train_mixed_precision=False,
    saving_period=10,
    checkpoint_file=None,
    load_optimizer_from_this_state_dict=None,
    output_path=None,
    **model_extra_args,
):
    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = (
        single_eval_pos_gen
        if callable(single_eval_pos_gen)
        else lambda: single_eval_pos_gen
    )

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt

    dl = priordataloader_class(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=bptt + (bptt_extra_samples if bptt_extra_samples else 0),
        device=device,
        **extra_prior_kwargs_dict,
    )

    encoder = encoder_generator(dl.num_features, emsize)
    style_def = next(iter(dl))[0][
        0
    ]  # This is (style, x, y), target with x and y with batch size
    print(f"Style definition: {style_def}")
    style_encoder = (
        style_encoder_generator(hyperparameter_definitions=style_def[0], em_size=emsize)
        if (style_def is not None)
        else None
    )
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif (
        isinstance(criterion, BarDistribution)
        or "BarDistribution" in criterion.__class__.__name__
    ):  # TODO remove this fix (only for dev)
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1
    model = TransformerModel(
        encoder,
        n_out,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        style_encoder=style_encoder,
        y_encoder=y_encoder_generator(1, emsize),
        input_normalization=input_normalization,
        pos_encoder=(
            pos_encoder_generator or positional_encodings.NoPositionalEncoding
        )(emsize, bptt * 2),
        decoder=decoder,
        init_method=initializer,
        **model_extra_args,
    )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    try:
        for (k, v), (k2, v2) in zip(
            model.state_dict().items(), initialize_with_model.state_dict().items()
        ):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, broadcast_buffers=False
        )

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(
        optimizer, warmup_epochs, epochs if epochs is not None else 100
    )  # when training for fixed time lr schedule takes 100 steps

    if load_optimizer_from_this_state_dict is not None:
        optimizer.load_state_dict(load_optimizer_from_this_state_dict)
    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.0
        total_positional_losses = 0.0
        total_positional_losses_recorded = 0
        before_get_batch = time.time()
        assert (
            len(dl) % aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            if using_dist and not (
                batch % aggregate_k_gradients == aggregate_k_gradients - 1
            ):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()

                with autocast(enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    output = model(
                        tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                        if isinstance(data, tuple)
                        else data.to(device),
                        single_eval_pos=single_eval_pos,
                    )

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert (
                            output.shape[-1] == 2
                        ), "need to write a little bit of code to handle multiple regression targets at once"

                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(
                            mean_pred.flatten(),
                            targets.to(device).flatten(),
                            var=var_pred.flatten(),
                        )
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        losses = criterion(
                            output.flatten(), targets.to(device).flatten()
                        )
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        losses = criterion(
                            output.reshape(-1, n_out),
                            targets.to(device).long().flatten(),
                        )
                    else:
                        losses = criterion(output, targets)
                    losses = losses.view(*output.shape[0:2])
                    loss = losses.mean() / aggregate_k_gradients

                if scaler:
                    loss = scaler.scale(loss)
                loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    try:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += losses.mean().cpu().detach()
                    total_positional_losses += (
                        losses.mean(1).cpu().detach()
                        if single_eval_pos is None
                        else nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                        * losses[: bptt - single_eval_pos].mean().cpu().detach()
                    )

                    total_positional_losses_recorded += (
                        torch.ones(bptt)
                        if single_eval_pos is None
                        else nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                    )

            before_get_batch = time.time()
        return (
            total_loss / steps_per_epoch,
            (total_positional_losses / total_positional_losses_recorded).tolist(),
            time_to_get_batch,
            forward_time,
            step_time,
        )

    total_loss = float("inf")
    total_positional_losses = float("inf")
    list_losses = []
    try:
        for epoch in range(1, epochs + 1) if epochs is not None else itertools.count(1):

            epoch_start_time = time.time()
            (
                total_loss,
                total_positional_losses,
                time_to_get_batch,
                forward_time,
                step_time,
            ) = train_epoch()
            list_losses.append(total_loss.item())
            if hasattr(dl, "validate") and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)

            else:
                val_score = None

            if epoch % saving_period == 0 and checkpoint_file is not None:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, checkpoint_file)
                full_model_path = checkpoint_file.split(".")[0] + "_full_model.pt"
                torch.save(model, full_model_path)

            if verbose:
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | "
                    f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f" data time {time_to_get_batch:5.2f} step time {step_time:5.2f}"
                    f" forward time {forward_time:5.2f}"
                    + (f"val score {val_score}" if val_score is not None else "")
                )
                print("-" * 89)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs)
            scheduler.step()
    except KeyboardInterrupt:
        pass

    if rank == 0:  # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        if output_path is not None:
            torch.save(model.to("cpu"), output_path)
            print("Checkpoint stored at ", output_path)
        return total_loss, total_positional_losses, model.to("cpu"), dl


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(
        description="Only used as a first parser for the config file path."
    )
    config_parser.add_argument("--config")
    parser = argparse.ArgumentParser()
    parser.add_argument("prior")
    parser.add_argument("--loss_function", default="barnll")
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument(
        "--min_y",
        type=float,
        help="barnll can only model y in strict ranges, this is the minimum y can take.",
    )
    parser.add_argument(
        "--max_y",
        type=float,
        help="barnll can only model y in strict ranges, this is the maximum y can take.",
    )
    parser.add_argument("--num_buckets", default=100, type=int)
    # parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior.')
    parser.add_argument(
        "--extra_prior_kwargs_dict",
        default={},
        dest="extra_prior_kwargs_dict",
        action=StoreDictKeyPair,
        nargs="+",
        metavar="KEY=VAL",
        help="Specify depending on the prior.",
    )
    parser.add_argument(
        "--encoder", default="linear", type=str, help="Specify depending on the prior."
    )
    parser.add_argument(
        "--y_encoder",
        default="linear",
        type=str,
        help="Specify depending on the prior. You should specify this if you do not fuse x and y.",
    )
    parser.add_argument(
        "--pos_encoder",
        default="none",
        type=str,
        help="Specify depending on the prior.",
    )
    parser.add_argument("--bptt", default=10, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=50, type=int)
    parser.add_argument("--validation_period", default=10, type=int)
    parser.add_argument(
        "--permutation_invariant_max_eval_pos",
        default=None,
        type=int,
        help="Set this to an int to ",
    )
    parser.add_argument(
        "--permutation_invariant_sampling",
        default="weighted",
        help="Only relevant if --permutation_invariant_max_eval_pos is set.",
    )
    parser.add_argument("--train_mixed_precision", action="store_true")

    # these can likely be mostly left at defaults
    parser.add_argument(
        "--emsize", default=512, type=int
    )  # sometimes even larger is better e.g. 1024
    parser.add_argument("--nlayers", default=6, type=int)
    parser.add_argument("--nhid", default=None, type=int)  # 2*emsize is the default
    parser.add_argument(
        "--nhead", default=4, type=int
    )  # nhead = emsize / 64 in the original paper
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--steps_per_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument(
        "--lr", "--learning_rate", default=0.001, type=float
    )  # try also .0003, .0001, go lower with lower batch size
    parser.add_argument("--gpu_device", default="cuda", type=str)

    # for model checkpointing
    parser.add_argument(
        "--checkpoint_file",
        help="absolute or relative-to-the-project-rootdir path to the file storing the state dicts.",
        default=None,
        type=str,
    )
    parser.add_argument("--saving_period", default=10, type=str)

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2 * args.emsize

    prior = args.__dict__.pop("prior")

    if prior == "gp":
        prior = priors.fast_gp.DataLoader
    elif prior == "ridge":
        prior = priors.ridge.DataLoader
    elif prior == "stroke":
        prior = priors.stroke.DataLoader
    elif prior == "mix_gp":
        prior = priors.fast_gp_mix.DataLoader
    else:
        raise NotImplementedError(f"Prior == {prior}.")

    loss_function = args.__dict__.pop("loss_function")

    criterion = nn.GaussianNLLLoss(reduction="none", full=True)
    classificiation_criterion = nn.CrossEntropyLoss(reduction="none")
    num_buckets = args.__dict__.pop("num_buckets")
    max_y = args.__dict__.pop("max_y")
    min_y = args.__dict__.pop("min_y")
    # criterion = nn.MSELoss(reduction='none')

    device = args.gpu_device if torch.cuda.is_available() else "cpu:0"

    def get_y_sample():
        args.__dict__["extra_prior_kwargs_dict"]["eval_pos_seq_len_sampler"] = lambda: (
            args.bptt,
            args.bptt,
        )
        dl = prior(
            num_steps=1,
            batch_size=args.batch_size * args.steps_per_epoch,
            seq_len=args.bptt,
            device=device,
            **args.extra_prior_kwargs_dict,
        )
        args.__dict__["extra_prior_kwargs_dict"].pop("eval_pos_seq_len_sampler")

        y_sample = next(iter(dl))[-2]
        print(
            f"Creating Bar distribution with borders from y sample of size {y_sample.numel()}"
        )
        return y_sample

    if loss_function == "ce":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif loss_function == "gaussnll":
        criterion = nn.GaussianNLLLoss(reduction="none", full=True)
    elif loss_function == "mse":
        criterion = nn.MSELoss(reduction="none")
    elif loss_function == "barnll":
        criterion = BarDistribution(
            borders=get_bucket_limits(num_buckets, full_range=(min_y, max_y))
        )
    elif loss_function == "adaptivebarnll":
        borders = get_bucket_limits(
            num_buckets, ys=get_y_sample(), full_range=(min_y, max_y)
        )
        criterion = BarDistribution(borders=borders)
    elif loss_function == "adaptivefullsupportbarnll":
        assert (
            min_y is None and max_y is None
        ), "Please do not specify `min_y` and `max_y` with `unboundedadaptivebarnll`."
        borders = get_bucket_limits(num_buckets, ys=get_y_sample())
        criterion = FullSupportBarDistribution(borders=borders)
    else:
        raise NotImplementedError(f"loss_function == {loss_function}.")

    encoder = args.__dict__.pop("encoder")
    y_encoder = args.__dict__.pop("y_encoder")

    def get_encoder_generator(encoder):
        if encoder == "linear":
            encoder_generator = encoders.Linear
        elif encoder == "mlp":
            encoder_generator = encoders.MLP
        elif encoder == "positional":
            encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f"A {encoder} encoder is not valid.")
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.__dict__.pop("pos_encoder")

    if pos_encoder == "none":
        pos_encoder_generator = None
    elif pos_encoder == "sinus":
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == "learned":
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == "paired_scrambled_learned":
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f"pos_encoer == {pos_encoder} is not valid.")

    permutation_invariant_max_eval_pos = args.__dict__.pop(
        "permutation_invariant_max_eval_pos"
    )
    permutation_invariant_sampling = args.__dict__.pop("permutation_invariant_sampling")
    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == "weighted":
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == "uniform":
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
        args.__dict__["single_eval_pos_gen"] = get_sampler(
            permutation_invariant_max_eval_pos
        )

    print("ARGS for `train`:", args.__dict__)

    if args.__dict__["checkpoint_file"] is not None:
        rootdir = os.path.dirname(os.path.realpath(__file__))
        args.__dict__["checkpoint_file"] = os.path.join(
            rootdir, args.__dict__["checkpoint_file"]
        )

        if os.path.exists(args.__dict__["checkpoint_file"]):
            state_dicts = torch.load(args.__dict__["checkpoint_file"])
            args.__dict__["load_weights_from_this_state_dict"] = state_dicts[
                "model_state_dict"
            ]
            args.__dict__["load_optimizer_from_this_state_dict"] = state_dicts[
                "optimizer_state_dict"
            ]
        else:
            args.__dict__["load_weights_from_this_state_dict"] = None
            args.__dict__["load_optimizer_from_this_state_dict"] = None

    train(
        prior,
        criterion,
        encoder_generator,
        y_encoder_generator=y_encoder_generator,
        pos_encoder_generator=pos_encoder_generator,
        **args.__dict__,
    )
