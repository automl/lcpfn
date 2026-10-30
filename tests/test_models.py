import os
import unittest

import torch

import lcpfn
from lcpfn.model import LCPFN


class TestModelRegistry(unittest.TestCase):
    def test_all_variants_registered(self):
        full = lcpfn.list_models()
        self.assertEqual(len(full), 3 * 3 * 3 * 2)
        for name in full:
            self.assertTrue(name.startswith("EPOCH"))
            self.assertTrue(lcpfn.model_dict[name].endswith(f"pfn_{name}.pt"))

    def test_short_names_point_to_1000_epochs(self):
        self.assertEqual(
            lcpfn.model_dict["EMSIZE512_NLAYERS12_NBUCKETS1000"],
            lcpfn.model_dict["EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000"],
        )
        self.assertIn(lcpfn.DEFAULT_MODEL, lcpfn.model_dict)
        self.assertEqual(len(lcpfn.model_dict), 54 + 18)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            lcpfn.download_model("EPOCH7_EMSIZE1_NLAYERS1_NBUCKETS1")
        with self.assertRaises(ValueError):
            LCPFN("no-such-model")


class TestSmallVariant(unittest.TestCase):
    """Downloads the smallest released variant (2.8 MB) and runs inference with it."""

    name = "EPOCH10_EMSIZE128_NLAYERS3_NBUCKETS1000"

    def test_download_and_predict(self):
        path = lcpfn.download_model(self.name)
        self.assertTrue(os.path.exists(path))
        model = LCPFN(self.name)
        x_train = torch.arange(1, 11).unsqueeze(-1)
        y_train = torch.linspace(0.1, 0.5, 10).unsqueeze(-1)
        x_test = torch.arange(11, 16).unsqueeze(-1)
        mean = model.predict_mean(x_train, y_train, x_test)
        self.assertEqual(mean.shape, (5, 1))
        self.assertTrue(torch.all(torch.isfinite(mean)))

    def test_getattr_compat(self):
        self.assertEqual(getattr(lcpfn, self.name), lcpfn.model_dict[self.name])


if __name__ == "__main__":
    unittest.main()
