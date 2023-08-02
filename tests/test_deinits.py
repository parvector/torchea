import unittest
from torch import nn
from torchea import deinits
from torchea.base import BaseIndvdL, BaseEA


class TestDeinits(unittest.TestCase):

    def test_uniform(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,3))
        ti.parameters_zero()
        for params in ti.parameters():
            self.assertEqual( (params==0).all(), True )

        bea = BaseEA()
        bea.register("init", deinits.uniform)
        bea.init(model=ti)
        for params in ti.parameters():
            self.assertEqual( (params!=0).all(), True )

    def test_normal(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,3))
        ti.parameters_zero()
        for params in ti.parameters():
            self.assertEqual( (params==0).all(), True )

        bea = BaseEA()
        bea.register("init", deinits.normal)
        bea.init(model=ti)
        for params in ti.parameters():
            self.assertEqual( (params!=0).all(), True )

    def test_constant(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,3))
        ti.parameters_zero()
        for params in ti.parameters():
            self.assertEqual( (params==0).all(), True )

        bea = BaseEA()
        bea.register("init", deinits.constant)
        bea.init(model=ti, val=2)
        for params in ti.parameters():
            self.assertEqual( (params==2).all(), True )

    def test_ones(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,3))

        bea = BaseEA()
        bea.register("init", deinits.ones)
        bea.init(model=ti)
        for params in ti.parameters():
            self.assertEqual( (params==1).all(), True )

    def test_zeros(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,3))

        bea = BaseEA()
        bea.register("init", deinits.zeros)
        bea.init(model=ti)
        for params in ti.parameters():
            self.assertEqual( (params==0).all(), True )

    def test_trunc_normal(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,3))
        ti.parameters_zero()
        for params in ti.parameters():
            self.assertEqual( (params==0).all(), True )

        bea = BaseEA()
        bea.register("init", deinits.trunc_normal)
        bea.init(model=ti)
        for params in ti.parameters():
            self.assertEqual( (params!=0).all(), True )

if __name__ == "__main__":    
    unittest.main()