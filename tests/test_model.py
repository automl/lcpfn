import unittest
import torch
from lcpfn.model import LCPFN

class TestLCPFN(unittest.TestCase):
    def setUp(self):
        self.model = LCPFN()

    def test_init(self):
        self.assertIsInstance(self.model, LCPFN)

    def test_predict_mean(self):
        x_train = torch.arange(1, 11).unsqueeze(-1)
        y_train = torch.rand(10).unsqueeze(-1)
        x_test = torch.arange(11, 16).unsqueeze(-1)
        mean = self.model.predict_mean(x_train, y_train, x_test)
        self.assertIsInstance(mean, torch.Tensor)
    
    def test_predict_quantiles(self):
        x_train = torch.arange(1, 11).unsqueeze(-1)
        y_train = torch.rand(10).unsqueeze(-1)
        x_test = torch.arange(11, 16).unsqueeze(-1)
        qs = [0.1, 0.5, 0.9]
        quantiles = self.model.predict_quantiles(x_train, y_train, x_test, qs)
        self.assertTrue(torch.all(quantiles[0] < quantiles[1]))
        self.assertTrue(torch.all(quantiles[1] < quantiles[2]))

if __name__ == '__main__':
    unittest.main()