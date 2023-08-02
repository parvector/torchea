import unittest
from torchea.base import BaseIndvdL
from torchea import demuts
from torch import nn



class TestDemuts(unittest.TestCase):

    def test_DE0(self):
        ti1 = BaseIndvdL()
        ti1.append(nn.Linear(2,2))
        ti1.parameters_zero()
        for params in ti1.parameters():
            params+=2

        ti2 = BaseIndvdL()
        ti2.append(nn.Linear(2,2))
        ti2.parameters_zero()
        for params in ti2.parameters():
            params+=2

        ti3 = BaseIndvdL()
        ti3.append(nn.Linear(2,2))
        ti3.parameters_zero()
        for params in ti3.parameters():
            params+=1

        #test mutDE0
        mut_model = demuts.DE0(ti1, ti2, ti3, 2)   
        for mutv in mut_model.parameters():
            self.assertTrue((mutv==4).all().item())