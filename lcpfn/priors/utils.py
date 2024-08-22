from lcpfn.utils import set_locals_in_self
from .prior import PriorDataLoader
import math


def get_batch_to_dataloader(get_batch_method_):
    class DL(PriorDataLoader):
        get_batch_method = get_batch_method_

        # Caution, you might need to set self.num_features manually if it is not part of the args.
        def __init__(self, num_steps, **get_batch_kwargs):
            set_locals_in_self(locals())

            # The stuff outside the or is set as class attribute before instantiation.
            self.num_features = (
                get_batch_kwargs.get("num_features") or self.num_features
            )
            print("DataLoader.__dict__", self.__dict__)

        @staticmethod
        def gbm(*args, eval_pos_seq_len_sampler, **kwargs):
            kwargs["single_eval_pos"], kwargs["seq_len"] = eval_pos_seq_len_sampler()
            # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
            # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
            if "dynamic_batch_size" in kwargs and kwargs["dynamic_batch_size"] > 0:
                kwargs["batch_size"] = kwargs["batch_size"] * math.floor(
                    math.pow(kwargs["seq_len_maximum"], kwargs["dynamic_batch_size"])
                    / math.pow(kwargs["seq_len"], kwargs["dynamic_batch_size"])
                )
            batch = get_batch_method_(*args, **kwargs)
            x, y, target_y, style = (
                batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
            )
            return (style, x, y), target_y, kwargs["single_eval_pos"]

        def __len__(self):
            return self.num_steps

        def __iter__(self):
            return iter(
                self.gbm(**self.get_batch_kwargs) for _ in range(self.num_steps)
            )

    return DL
