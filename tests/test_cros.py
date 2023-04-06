import unittest
from torchea import cros
from torch import nn
from torchea.base import BaseIndvdl



class TestCros(unittest.TestCase):
    def test_crosDE0(self):
        ti0 = BaseIndvdl()
        ti0.append(nn.Linear(2,2))
        ti0.parameters_zero()
        for params in ti0.parameters():
            params += 0

        ti1 = BaseIndvdl()
        ti1.append(nn.Linear(2,2))
        ti1.parameters_zero()
        for params in ti1.parameters():
            params += 1

        modelu = cros.crosDE0(ti0, ti1, CR=1)
        for params in modelu.parameters():
            self.assertEqual((params.flatten()==1).all(), True)

if __name__ == "__main__":    
    unittest.main()