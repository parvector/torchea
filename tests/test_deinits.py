import unittest
from torch import nn
from torchea import deinits
from torchea.base import IndvdL, BaseEA


class TestDeinits(unittest.TestCase):

    def test_uniform(self):
        ti = IndvdL(modules=[nn.Linear(2,3), nn.Linear(3,4)])
        ti.setarget()
        ti.parameters_zero()

        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.uniform)
        bea.init(model=ti)
        for params in ti.parameters():
            self.assertEqual( (params!=0).all(), True )
            self.assertEqual( (params>0).all(), True )
            self.assertEqual( (params<1).all(), True )


        ti.parameters_zero()
        ti.untarget(mindxs=[1])

        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.uniform)
        bea.init(model=ti)
        for mi, module in enumerate(ti):
            if mi == 0:
                for param in module.parameters():
                    self.assertEqual( (param!=0).all(), True )
                    self.assertEqual( (param>0).all(), True )
                    self.assertEqual( (param<1).all(), True )
            elif mi == 1:
                for param in module.parameters():
                    self.assertEqual( (param==0).all(), True )

    def test_normal(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,4))
        ti.setarget()
        ti.parameters_zero()
        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.normal)

        bea.init(model=ti)
        for param in ti.parameters():
            self.assertEqual( 1.7 > param.std() > -1.7, True )
            self.assertEqual( 1.7 > param.mean() > -1.7, True )

        ti.parameters_zero()
        ti.untarget(mindxs=[1])
        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.normal)
        bea.init(model=ti)
        for mi, module in enumerate(ti):
            if mi == 0:
                for param in module.parameters():
                    self.assertEqual( 1.7 > param.std() > -1.7, True )
                    self.assertEqual( 1.7 > param.mean() > -1.7, True )
            elif mi == 1:
                for param in module.parameters():
                    self.assertEqual( (param==0).all(), True )

    def test_constant(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,4))
        ti.setarget()
        ti.parameters_zero()
        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.constant)

        bea.init(model=ti, val=2)
        for param in ti.parameters():
            self.assertEqual( (param==2).all(), True )


        ti.parameters_zero()
        ti.untarget(mindxs=[1])

        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.constant)
        bea.init(model=ti, val=2)
        for mi, module in enumerate(ti):
            if mi == 0:
                for param in module.parameters():
                    self.assertEqual( (param==2).all(), True )

            elif mi == 1:
                for param in module.parameters():
                    self.assertEqual( (param==0).all(), True )

    def test_ones(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,4))
        ti.setarget()
        ti.parameters_zero()
        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.ones)

        bea.init(model=ti)
        for param in ti.parameters():
            self.assertEqual( (param==1).all(), True )


        ti.parameters_zero()
        ti.untarget(mindxs=[1])

        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.ones)
        bea.init(model=ti)
        for mi, module in enumerate(ti):
            if mi == 0:
                for param in module.parameters():
                    self.assertEqual( (param==1).all(), True )

            elif mi == 1:
                for param in module.parameters():
                    self.assertEqual( (param==0).all(), True )

    def test_zeros(self):
        ti = IndvdL([nn.Linear(2,3)])
        ti.setarget()

        bea = BaseEA(src_indvd=ti)
        bea.register("init", deinits.zeros)
        bea.init(model=ti)
        for params in ti.parameters():
            if params.target_torchea:
                self.assertEqual( (params==0).all(), True )
            else:
                self.assertEqual( (params!=0).all(), True )


if __name__ == "__main__":    
    unittest.main()