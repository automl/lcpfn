import gzip
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))


model_path = "trained_models"
MODEL_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/lcpfn/"

# All LC-PFN variants released with the paper (Table 2 and the appendix):
# training epochs x embedding size x number of transformer layers x output buckets.
EPOCHS = (10, 100, 1000)
EMSIZES = (128, 256, 512)
NLAYERS = (3, 6, 12)
NBUCKETS = (1000, 10000)

DEFAULT_MODEL = "EMSIZE512_NLAYERS12_NBUCKETS1000"


def _model_filename(epoch, emsize, nlayers, nbuckets):
    return f"pfn_EPOCH{epoch}_EMSIZE{emsize}_NLAYERS{nlayers}_NBUCKETS{nbuckets}.pt"


def _model_abspath(filename):
    return os.path.join(os.path.dirname(__file__), model_path, filename)


# model_dict maps a model name to the local path of its weights.
# Full names: EPOCH{10,100,1000}_EMSIZE{128,256,512}_NLAYERS{3,6,12}_NBUCKETS{1000,10000}.
# Short names without EPOCH (e.g. the default "EMSIZE512_NLAYERS12_NBUCKETS1000") refer to the
# fully trained 1000-epoch models and are kept for backward compatibility.
model_dict = {}
for _e in EPOCHS:
    for _s in EMSIZES:
        for _l in NLAYERS:
            for _b in NBUCKETS:
                model_dict[f"EPOCH{_e}_EMSIZE{_s}_NLAYERS{_l}_NBUCKETS{_b}"] = _model_abspath(
                    _model_filename(_e, _s, _l, _b)
                )
for _s in EMSIZES:
    for _l in NLAYERS:
        for _b in NBUCKETS:
            model_dict[f"EMSIZE{_s}_NLAYERS{_l}_NBUCKETS{_b}"] = model_dict[
                f"EPOCH1000_EMSIZE{_s}_NLAYERS{_l}_NBUCKETS{_b}"
            ]
del _e, _s, _l, _b

# Between 2024-08-22 and 2026-10-30 the file served under the 6-layer name was a copy of the
# 12-layer weights. If a previously downloaded 6-layer file has that content, download it again.
_KNOWN_BAD_SHA256 = {
    "pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000.pt": "195df6bafff54addf3535ceb15944d1f6d48924221911f7b48517dab2e0e7b84",
}


def list_models():
    """Names of all 54 released LC-PFN variants (full names, sorted)."""
    return sorted(k for k in model_dict if k.startswith("EPOCH"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_known_bad(weights_path):
    expected_bad = _KNOWN_BAD_SHA256.get(os.path.basename(weights_path))
    return expected_bad is not None and _sha256(weights_path) == expected_bad


def download_model(name, force=False):
    """Make sure the weights of model `name` are available locally and return their path.

    `name` is a key of `model_dict` (see `list_models()`). The gzipped checkpoint is fetched from
    MODEL_URL into the package's `trained_models` folder and decompressed. Nothing is downloaded if
    the weights are already present, unless `force` is set.
    """
    if name not in model_dict:
        raise ValueError(
            f"Unknown LC-PFN model {name!r}. Available models: {', '.join(list_models())}"
        )
    weights_path = model_dict[name]
    filename = os.path.basename(weights_path)
    compressed_weights_path = weights_path + ".gz"

    if os.path.exists(weights_path) and not force and _is_known_bad(weights_path):
        print(f"{filename} on disk is a known-bad download (12-layer weights under the 6-layer name); re-downloading.")
        force = True
    if force:
        for p in (weights_path, compressed_weights_path):
            if os.path.exists(p):
                os.remove(p)

    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        if not os.path.exists(compressed_weights_path):
            import requests

            url = MODEL_URL + filename + ".gz"
            print("Downloading", url, "to", os.path.abspath(compressed_weights_path))
            tmp = compressed_weights_path + ".part"
            with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(tmp, compressed_weights_path)
        print("Unzipping", filename)
        tmp = weights_path + ".part"
        with gzip.open(compressed_weights_path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, weights_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Failed to obtain {weights_path}. Make sure you have an internet connection to download the model automatically."
        )
    return weights_path


def prepare_models(model_names=None):
    """Download the given models (default: the two models shipped with lcpfn<=0.1.3)."""
    if model_names is None:
        model_names = [DEFAULT_MODEL, "EMSIZE512_NLAYERS6_NBUCKETS1000"]
    for name in model_names:
        print("Successfully located model at", download_model(name))


def __getattr__(name):
    if name in model_dict:
        return download_model(name)
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
    "model_dict",
    "list_models",
    "download_model",
    "prepare_models",
    "DEFAULT_MODEL",
    "__version__",
]
