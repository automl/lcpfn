import torch
import lcpfn
import warnings
from lcpfn import utils


class LCPFN(torch.nn.Module):
    def __init__(self, model_name="EMSIZE512_NLAYERS12_NBUCKETS1000"):
        super(LCPFN, self).__init__()
        self.model = torch.load(
            getattr(lcpfn, model_name) if model_name in lcpfn.model_dict else model_name
        )
        self.model.eval()

    def check_input(self, x_train, x_test, y_train, y_test=None):
        if torch.any(x_train < 0) or torch.any(x_test < 0):
            # raise warning if input has negative values
            raise Exception("x values should be non-negative")
        if torch.any((0 > y_train) | (y_train > 1)) or (
            y_test is not None and torch.any(0 < y_test < 1)
        ):
            # raise warning if input has values outside [0,1]
            raise Exception(
                "y values should be in the range [0,1]. Please set normalizer_kwargs accordingly."
            )

    @torch.no_grad()
    def predict_mean(
        self, x_train, y_train, x_test, normalizer=utils.identity_normalizer()
    ):
        y_train_norm = normalizer[0](y_train)
        logits = self(x_train=x_train, y_train=y_train_norm, x_test=x_test)
        return normalizer[1](self.model.criterion.mean(logits))

    @torch.no_grad()
    def predict_quantiles(
        self, x_train, y_train, x_test, qs, normalizer=utils.identity_normalizer()
    ):
        y_train_norm = normalizer[0](y_train)
        logits = self(x_train=x_train, y_train=y_train_norm, x_test=x_test)
        return normalizer[1](
            torch.cat([self.model.criterion.icdf(logits, q) for q in qs], dim=1)
        )

    @torch.no_grad()
    def nll_loss(self, x_train, y_train, x_test, y_test):
        # TODO add normalizer_kwargs
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion(logits, y_test)

    def forward(self, x_train, y_train, x_test):
        self.check_input(x_train, x_test, y_train)
        single_eval_pos = x_train.shape[0]
        x = torch.cat([x_train, x_test], dim=0).unsqueeze(1)
        y = y_train.unsqueeze(1)
        return self.model((x, y), single_eval_pos=single_eval_pos)
