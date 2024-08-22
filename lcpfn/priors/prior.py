from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader


class PriorDataLoader(DataLoader, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        num_steps,
        batch_size,
        eval_pos_seq_len_sampler,
        seq_len_maximum,
        device,
        **kwargs,
    ):
        """

        :param num_steps: int, first argument, the number of steps to take per epoch, i.e. iteration of the DataLoader
        :param batch_size: int, number of datasets per batch
        :param eval_pos_seq_len_sampler: callable, it takes no arguments and returns a tuple (single eval pos, bptt)
        :param kwargs: for future compatibility it is good to have a final all catch, as new kwargs might be introduced
        """
        pass

    # A class or object variable `num_features`: int
    # Optional: `validate` function that accepts a transformer model

    # The DataLoader iter should return batches of the form ([style], x, y), target_y, single_eval_pos
    # We follow sequence len (s) first, batch size (b) second. So x: (s,b,num_features), y,target_y: (s,b)
    # and style: Optional[(b,num_style_params)], style can be omitted or set to None, if it is not intended to be used.

    # For more references, see `priors/utils.py` for a pretty general implementation of a DataLoader
    # and `train.py` for the only call of it.
