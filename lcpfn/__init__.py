import os, sys

sys.path.insert(0, os.path.dirname(__file__))


model_path = "trained_models"


def prepare_models():
    pfns4bo_dir = os.path.dirname(__file__)
    model_names = [
        "pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt",
        "pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000.pt",
    ]

    for name in model_names:
        weights_path = os.path.join(pfns4bo_dir, model_path, name)
        compressed_weights_path = os.path.join(pfns4bo_dir, model_path, name + ".gz")
        if not os.path.exists(weights_path):
            if not os.path.exists(compressed_weights_path):
                print("Downloading", os.path.abspath(compressed_weights_path))
                import requests

                url = f'https://ml.informatik.uni-freiburg.de/research-artifacts/lcpfn/{name + ".gz"}'
                r = requests.get(url, allow_redirects=True)
                os.makedirs(os.path.dirname(compressed_weights_path), exist_ok=True)
                with open(compressed_weights_path, "wb") as f:
                    f.write(r.content)
            if os.path.exists(compressed_weights_path):
                print("Unzipping", name)
                os.system(f"gzip -dk {compressed_weights_path}")
            else:
                print("Failed to find", compressed_weights_path)
                print(
                    "Make sure you have an internet connection to download the model automatically.."
                )
        if os.path.exists(weights_path):
            print("Successfully located model at", weights_path)


model_dict = {
    "EMSIZE512_NLAYERS12_NBUCKETS1000": os.path.join(
        os.path.dirname(__file__),
        model_path,
        "pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt",
    ),
    "EMSIZE512_NLAYERS6_NBUCKETS1000": os.path.join(
        os.path.dirname(__file__),
        model_path,
        "pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000.pt",
    ),
}


def __getattr__(name):
    if name in model_dict:
        if not os.path.exists(model_dict[name]):
            print(
                "Can't find",
                os.path.abspath(model_dict[name]),
                "thus unzipping/downloading models now.",
            )
            print("This might take a while..")
            prepare_models()
        return model_dict[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


from .version import __version__
from lcpfn.model import LCPFN
from lcpfn.train_lcpfn import train_lcpfn
from lcpfn.domhan_prior import sample_from_prior, create_get_batch_func

__all__ = [
    "LCPFN",
    "train_lcpfn",
    "sample_from_prior",
    "create_get_batch_func",
    "__version__",
]
