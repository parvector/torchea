import unittest
from torchea import cros
from torch import nn
import torch
from torchea.base import BaseIndvdl



class TestCros(unittest.TestCase):
    def test_crosDE0(self):
        modelx = BaseIndvdl()
        modelx.append(nn.Linear(4,2))
        modelx.parameters_zero()

        modelv = BaseIndvdl()
        modelv.append(nn.Linear(4,2))
        modelv.parameters_zero()
        for params in modelv.parameters():
            params += 1

        modelu = cros.crosDE0(modelx, modelv, CR=0)
        for params in modelu.parameters():
            self.assertEqual((params.flatten()==0).all(), True)
        
        sum_ones = 0
        count_tests = 500
        min_treshold = 0.8
        max_treshold = 1.2
        for i in range(count_tests):
            modelu = cros.crosDE0(modelx, modelv, CR=0.1)
            count_ones = sum([ (params.data == 1).sum() for params in modelu.parameters() ]).item()
            sum_ones += count_ones
        avg_value = sum_ones/count_tests
        self.assertEqual(min_treshold<avg_value<max_treshold, True)
        

        sum_ones = 0
        count_tests = 500
        min_treshold = 4.8
        max_treshold = 5.2
        for i in range(count_tests):
            modelu = cros.crosDE0(modelx, modelv, CR=0.5)
            count_ones = sum([ (params.data == 1).sum() for params in modelu.parameters() ]).item()
            sum_ones += count_ones
        avg_value = sum_ones/count_tests
        self.assertEqual(min_treshold<avg_value<max_treshold, True)

        sum_ones = 0
        count_tests = 500
        min_treshold = 8.8
        max_treshold = 9.2
        for i in range(count_tests):
            modelu = cros.crosDE0(modelx, modelv, CR=0.9)
            count_ones = sum([ (params.data == 1).sum() for params in modelu.parameters() ]).item()
            sum_ones += count_ones
        avg_value = sum_ones/count_tests
        self.assertEqual(min_treshold<avg_value<max_treshold, True)

        modelu = cros.crosDE0(modelx, modelv, CR=1)
        for params in modelu.parameters():
            self.assertEqual((params.flatten()==1).all(), True)

if __name__ == "__main__":    
    unittest.main()