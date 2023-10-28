import torch
import lcpfn

class LCPFN(torch.nn.Module):
    def __init__(self, model_name="EMSIZE512_NLAYERS12_NBUCKETS1000"):
        super(LCPFN, self).__init__()
        self.model = torch.load(getattr(lcpfn, model_name) if model_name in lcpfn.model_dict else model_name)
        self.model.eval()

    @torch.no_grad()
    def predict_mean(self, x_train, y_train, x_test):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion.mean(logits)

    @torch.no_grad()
    def predict_quantiles(self, x_train, y_train, x_test, qs):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return torch.cat([self.model.criterion.icdf(logits, q) for q in qs], dim=1)

    @torch.no_grad()
    def nll_loss(self, x_train, y_train, x_test, y_test):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion(logits, y_test)

    def forward(self, x_train, y_train, x_test):
        single_eval_pos = x_train.shape[0]
        x = torch.cat([x_train, x_test], dim=0).unsqueeze(1)
        y = y_train.unsqueeze(1)
        return self.model((x, y), single_eval_pos=single_eval_pos)