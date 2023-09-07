import unittest
from torchea.base import IndvdL, BaseEA
from torchea import demuts
from torch import nn
import torch


class TestDemuts(unittest.TestCase):

    def test_mut0(self):
        ti1 = IndvdL()
        ti1.append(nn.Linear(2,2))
        ti1.setarget()
        ti1.parameters_zero()
        with torch.no_grad():
            for params in ti1.parameters():
                params+=2

        ti2 = IndvdL()
        ti2.append(nn.Linear(2,2))
        ti2.setarget()
        ti2.parameters_zero()
        with torch.no_grad():
            for params in ti2.parameters():
                params+=2

        ti3 = IndvdL()
        ti3.append(nn.Linear(2,2))
        ti3.setarget()
        ti3.parameters_zero()
        with torch.no_grad():
            for params in ti3.parameters():
                params+=1

        #test mut0
        tea = BaseEA(src_indvd=ti1)
        tea.register('mut', demuts.mut0)
        mut_model = tea.mut(ti1, ti2, ti3, 2)   
        for mutv in mut_model.parameters():
            self.assertTrue((mutv==4).all().item())