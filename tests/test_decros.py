import unittest
from torchea import decros
from torch import nn
import torch
from torchea.base import BaseIndvdL, BaseEA



class TestDecros(unittest.TestCase):
    def test_cros0(self):
        modelx = BaseIndvdL()
        modelx.append(nn.Linear(4,2))
        modelx.setarget()
        modelx.parameters_zero()

        modelv = BaseIndvdL()
        modelv.append(nn.Linear(4,2))
        modelv.setarget()
        modelv.parameters_zero()
        with torch.no_grad():
            for params in modelv.parameters():
                params += 1

        bea = BaseEA(src_indvd=None)
        bea.register("cros", decros.cros0)
        modelu=bea.cros(modelx, modelv, CR=0)
        for params in modelu.parameters():
            self.assertEqual((params.flatten()==0).all().item(), True)
        
        sum_ones = 0
        sum_not_ones = 0
        count_tests = 400
        min_treshold = 0.08
        max_treshold = 0.12
        for i in range(count_tests):
            modelu = bea.cros(modelx, modelv, CR=0.1)
            count_ones = sum([ (params.data == 1).sum() for params in modelu.parameters() ]).item()
            count_not_ones = sum([ (params.data!= 1).sum() for params in modelu.parameters() ]).item()
            sum_ones += count_ones
            sum_not_ones += count_not_ones
        sum_all = sum_ones+sum_not_ones
        self.assertEqual(min_treshold<sum_ones/sum_all<max_treshold, True)
        

        sum_ones = 0
        sum_not_ones = 0
        count_tests = 400
        min_treshold = 0.48
        max_treshold = 0.52
        for i in range(count_tests):
            modelu = bea.cros(modelx, modelv, CR=0.5)
            count_ones = sum([ (params.data == 1).sum() for params in modelu.parameters() ]).item()
            count_not_ones = sum([ (params.data!= 1).sum() for params in modelu.parameters() ]).item()
            sum_ones += count_ones
            sum_not_ones += count_not_ones
        sum_all = sum_ones+sum_not_ones
        self.assertEqual(min_treshold<sum_ones/sum_all<max_treshold, True)

        sum_ones = 0
        sum_not_ones = 0
        count_tests = 400
        min_treshold = 0.88
        max_treshold = 0.92
        for i in range(count_tests):
            modelu = bea.cros(modelx, modelv, CR=0.9)
            count_ones = sum([ (params.data == 1).sum() for params in modelu.parameters() ]).item()
            count_not_ones = sum([ (params.data!= 1).sum() for params in modelu.parameters() ]).item()
            sum_ones += count_ones
            sum_not_ones += count_not_ones
        sum_all = sum_ones+sum_not_ones
        self.assertEqual(min_treshold<sum_ones/sum_all<max_treshold, True)

        modelu = bea.cros(modelx, modelv, CR=1)
        for params in modelu.parameters():
            self.assertEqual((params.flatten()==1).all(), True)

if __name__ == "__main__":    
    unittest.main()