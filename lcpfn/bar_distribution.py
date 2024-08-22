import torch
from torch import nn


class BarDistribution(nn.Module):
    def __init__(
        self, borders: torch.Tensor, smoothing=0.0
    ):  # here borders should start with min and end with max, where all values lie in (min,max) and are sorted
        # sorted list of borders
        super().__init__()
        assert len(borders.shape) == 1
        # self.borders = borders
        self.register_buffer("borders", borders)
        self.register_buffer("smoothing", torch.tensor(smoothing))
        # self.bucket_widths = self.borders[1:] - self.borders[:-1]
        self.register_buffer("bucket_widths", self.borders[1:] - self.borders[:-1])
        full_width = self.bucket_widths.sum()
        border_order = torch.argsort(borders)
        assert (
            full_width - (self.borders[-1] - self.borders[0])
        ).abs() < 1e-4, f"diff: {full_width - (self.borders[-1] - self.borders[0])}"
        assert (
            border_order == torch.arange(len(borders)).to(border_order.device)
        ).all(), "Please provide sorted borders!"
        self.num_bars = len(borders) - 1

    def map_to_bucket_idx(self, y):
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def forward(
        self, logits, y
    ):  # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        target_sample = self.map_to_bucket_idx(y)
        assert (target_sample >= 0).all() and (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"

        bucket_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        # print(bucket_log_probs, logits.shape)

        nll_loss = -scaled_bucket_log_probs.gather(
            -1, target_sample.unsqueeze(-1)
        ).squeeze(-1)

        smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
        smoothing = self.smoothing if self.training else 0.0
        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        return loss

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def icdf(self, logits, left_prob):
        """
        Implementation of the quantile function
        :param logits: Tensor of any shape, with the last dimension being logits
        :param left_prob: float: The probability mass to the left of the result.
        :return: Position with `left_prob` probability weight to the left.
        """
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        idx = (
            torch.searchsorted(
                cumprobs,
                left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=probs.device),
            )
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )  # this might not do the right for outliers
        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1
        )

        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1, idx[..., None]
        ).squeeze(-1)

    def quantile(self, logits, center_prob=0.682):
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (self.icdf(logits, side_probs), self.icdf(logits, 1.0 - side_probs)), -1
        )

    def ucb(self, logits, best_f, rest_prob=(1 - 0.682) / 2, maximize=True):
        """
        UCB utility. Rest Prob is the amount of utility above (below) the confidence interval that is ignored.
        Higher rest_prob is equivalent to lower beta in the standard GP-UCB formulation.
        :param logits: Logits, as returned by the Transformer.
        :param best_f: Only here, since the other utilities have it.
        :param rest_prob: The amount of utility above (below) the confidence interval that is ignored.
        The default is equivalent to using GP-UCB with `beta=1`.
        To get the corresponding `beta`, where `beta` is from
        the standard GP definition of UCB `ucb_utility = mean + beta * std`,
        you can use this computation: `beta = math.sqrt(2)*torch.erfinv(torch.tensor(2*rest_prob-1))`.
        :param maximize:
        :return: utility
        """
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def mode(self, logits):
        mode_inds = logits.argmax(-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def ei(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        if maximize:
            bucket_contributions = torch.tensor(
                [
                    max((bucket_max + max(bucket_min, best_f)) / 2 - best_f, 0)
                    for bucket_min, bucket_max, bucket_mean in zip(
                        self.borders[:-1], self.borders[1:], bucket_means
                    )
                ],
                dtype=logits.dtype,
                device=logits.device,
            )
        else:
            bucket_contributions = torch.tensor(
                [
                    -min((min(bucket_max, best_f) + bucket_min) / 2 - best_f, 0)
                    for bucket_min, bucket_max, bucket_mean in zip(  # min on max instead of max on min, and compare min < instead of max >
                        self.borders[:-1], self.borders[1:], bucket_means
                    )
                ],
                dtype=logits.dtype,
                device=logits.device,
            )
        p = torch.softmax(logits, -1)
        return p @ bucket_contributions

    def pi(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        p = torch.softmax(logits, -1)
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1.0 - ((best_f - self.borders[:-1]) / border_widths).clamp(0.0, 1.0)
        return (p * factor).sum(-1)

    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits):
        return self.mean_of_square(logits) - self.mean(logits).square()


class FullSupportBarDistribution(BarDistribution):
    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p=0.5):
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p)
        )
        return torch.distributions.HalfNormal(s)

    def forward(
        self, logits, y
    ):  # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        assert self.num_bars > 1
        target_sample = self.map_to_bucket_idx(y)
        target_sample.clamp_(0, self.num_bars - 1)
        assert logits.shape[-1] == self.num_bars

        bucket_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        # print(bucket_log_probs, logits.shape)
        log_probs = scaled_bucket_log_probs.gather(
            -1, target_sample.unsqueeze(-1)
        ).squeeze(-1)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        # TODO look over it again
        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (self.borders[1] - y[target_sample == 0]).clamp(min=0.00000001)
        ) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars - 1] += side_normals[1].log_prob(
            y[target_sample == self.num_bars - 1] - self.borders[-2]
        ) + torch.log(self.bucket_widths[-1])

        nll_loss = -log_probs

        smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
        smoothing = self.smoothing if self.training else 0.0
        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss

        return loss

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means


def get_bucket_limits_(
    num_outputs: int,
    full_range: tuple = None,
    ys: torch.Tensor = None,
    verbose: bool = False,
):
    assert (ys is not None) or (full_range is not None)
    if ys is not None:
        ys = ys.flatten()
        if len(ys) % num_outputs:
            ys = ys[: -(len(ys) % num_outputs)]
        print(
            f"Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys."
        )
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min() and full_range[1] >= ys.max()
            full_range = torch.tensor(full_range)
        ys_sorted, ys_order = ys.sort(0)
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        if verbose:
            print(
                f"Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys."
            )
            print(full_range)
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)], 0
        )

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,
                torch.tensor(full_range[1]).unsqueeze(0),
            ],
            0,
        )

    assert (
        len(bucket_limits) - 1 == num_outputs
        and full_range[0] == bucket_limits[0]
        and full_range[-1] == bucket_limits[-1]
    )
    return bucket_limits


def get_bucket_limits(
    num_outputs: int,
    full_range: tuple = None,
    ys: torch.Tensor = None,
    verbose: bool = False,
):
    assert (ys is None) != (
        full_range is None
    ), "Either full_range or ys must be passed."

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        if len(ys) % num_outputs:
            ys = ys[: -(len(ys) % num_outputs)]
        print(
            f"Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys."
        )
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert (
                full_range[0] <= ys.min() and full_range[1] >= ys.max()
            ), f"full_range {full_range} not in range of ys {ys.min(), ys.max()}"
            full_range = torch.tensor(full_range)
        ys_sorted, ys_order = ys.sort(0)
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        if verbose:
            print(
                f"Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys."
            )
            print(full_range)
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)], 0
        )

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,
                torch.tensor(full_range[1]).unsqueeze(0),
            ],
            0,
        )

    assert (
        len(bucket_limits) - 1 == num_outputs
    ), f"len(bucket_limits) - 1 == {len(bucket_limits) - 1} != {num_outputs} == num_outputs"
    assert full_range[0] == bucket_limits[0], f"{full_range[0]} != {bucket_limits[0]}"
    assert (
        full_range[-1] == bucket_limits[-1]
    ), f"{full_range[-1]} != {bucket_limits[-1]}"

    return bucket_limits
