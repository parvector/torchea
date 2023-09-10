import unittest
import torch
from torch import nn
from torchea.base import BaseIndvd, IndvdL, IndvdD, BaseEA
from torchea.deinits import uniform
from datetime import datetime
import numpy as np



class TestBaseIndvd(unittest.TestCase):
    def test_init(self):
        ti = BaseIndvd()
        self.assertEqual(type(ti.birthtime), datetime)
        self.assertEqual(type(ti.name), str)

    def test_deepcopy(self):
        ti0 = IndvdL()
        ti0.append(nn.Linear(2,2))
        ti0.append(nn.Linear(2,3))
        ti0.append(nn.Linear(3,2))
        ti0.fitnes = (None, 1, 2)
        ti0.setarget()

        ti1 = ti0.deepcopy()
        self.assertEqual( id(ti0) != id(ti1), True)
        self.assertEqual( all([ ti0fitnes == ti1fitnes for ti0fitnes, ti1fitnes in zip(ti0.fitnes, ti1.fitnes)]), True)
        for ti0param, ti1param in zip(ti0.parameters(), ti1.parameters()):
            self.assertEqual(ti0param.target_torchea, ti1param.target_torchea)

class TestIndvdL(unittest.TestCase):
    def test_init(self):
        ti = IndvdL(modules=[nn.Linear(2,3), nn.Linear(3,4)])
        self.assertEqual(len(ti), 2)
        self.assertEqual(type(ti.birthtime), datetime)
        self.assertEqual(type(ti.name), str)

    def test_setarget(self):
        til = IndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.setarget(mindxs=[1])
        for mi, module in enumerate(til):
            for param in module.parameters():
                if mi == 1:
                    self.assertEqual(param.target_torchea, True)
                else:
                    self.assertEqual(param.target_torchea, False)


        til = IndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.setarget(mindxs=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.target_torchea, True)
            else:
                self.assertEqual(param.target_torchea, False)


        til = IndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.setarget()
        for param in til.parameters():
            self.assertEqual(param.target_torchea, True)

    def test_untarget(self):
        til = IndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.setarget()
        til.untarget(mindxs=[1])
        for mi, module in enumerate(til):
            for param in module.parameters():
                if mi == 1:
                    self.assertEqual(param.target_torchea, False)
                else:
                    self.assertEqual(param.target_torchea, True)


        til = IndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.setarget()
        til.untarget(mindxs=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.target_torchea, False)
            else:
                self.assertEqual(param.requires_grad, True)


        til = IndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.setarget()
        til.untarget()
        for param in til.parameters():
            self.assertEqual(param.target_torchea, False)

    def test_parameters_zero(self):
        ti = IndvdD(modules={"0":nn.Linear(2,2), "1":nn.Linear(2,3), "2":nn.Linear(3,2)})

        ti.setarget()
        for param in ti.parameters():
            self.assertEqual((param==0).all(), 0)

        ti = IndvdD(modules={"0":nn.Linear(2,2), "1":nn.Linear(2,3), "2":nn.Linear(3,2)})
        ti.untarget(mkeys=["0", "2"])
        for mk, module in ti.items():
            if mk in ["0", "2"]:
                for param in module.parameters():
                    self.assertEqual((param==0).all(), 0)
            else:
                for param in module.parameters():
                    self.assertEqual((param!=0).all(),True)

    def test_count_ws(self):
        ti = IndvdL(modules=[nn.Linear(2,2), nn.Linear(2,3), nn.Linear(3,2)])
        ti.setarget()

        self.assertEqual(23, ti.count_ws())

        ti.untarget(mindxs=[0,2])
        self.assertEqual(9, ti.count_ws())

        self.assertEqual(23, ti.count_ws(only_targets=False))

    def test_getv(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.setarget()
        ti.parameters_zero()

        # test raises with all target tensors
        ti.setarget()
        with self.assertRaises(IndexError):
            ti.getv(15)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test with all target tensors
        ti.setarget()
        params = list(ti.parameters())

        ti.parameters_zero()
        params[0].data[0,0] = 1
        self.assertEqual(1, ti.getv(0))

        ti.parameters_zero()
        params[0].data[0,1] = 1
        self.assertEqual(1, ti.getv(1))

        ti.parameters_zero()
        params[0].data[1,1] = 1
        self.assertEqual(1, ti.getv(3))


        ti.parameters_zero()
        params[1].data[0] = 1
        self.assertEqual(1, ti.getv(4))
        ti.parameters_zero()
        params[1].data[1] = 1
        self.assertEqual(1, ti.getv(5))

        ti.parameters_zero()
        params[2].data[0,0] = 1
        self.assertEqual(1, ti.getv(6))

        ti.parameters_zero()
        params[2].data[1,1] = 1
        self.assertEqual(1, ti.getv(9))

        ti.parameters_zero()
        params[2].data[2,0] = 1
        self.assertEqual(1, ti.getv(10))

        ti.parameters_zero()
        params[3].data[0] = 1
        self.assertEqual(1, ti.getv(12))

        ti.parameters_zero()
        params[3].data[1] = 1
        self.assertEqual(1, ti.getv(13))

        ti.parameters_zero()
        params[3].data[2] = 1
        self.assertEqual(1, ti.getv(14))

        # test raises without all target tensors
        target_tindxs = [1,2]
        ti.untarget()
        ti.setarget(mindxs=None, tindxs=target_tindxs)
        with self.assertRaises(IndexError):
            ti.getv(15, only_targets=False)
        with self.assertRaises(IndexError):
            ti.getv(8)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test without all target tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        params[0].data[0,0] = 1
        params[1].data[0] = 2
        self.assertEqual(1, ti.getv(0, only_targets=False))
        self.assertEqual(2, ti.getv(0))

        ti.parameters_zero()
        params[1].data[1] = 1
        params[2].data[1,1] = 2
        self.assertEqual(1, ti.getv(5, only_targets=False))
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        params[2].data[0,1] = 1
        params[2].data[2,1] = 2
        self.assertEqual(1, ti.getv(7, only_targets=False))
        self.assertEqual(2, ti.getv(7))


    def test_setv(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.setarget()
        ti.parameters_zero()

        # test raises with all target tensors
        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test with all target tensors
        ti.parameters_zero()
        ti.setv(0, 1)
        self.assertEqual(1, ti.getv(0))
        
        ti.parameters_zero()
        ti.setv(2, 1)
        self.assertEqual(1, ti.getv(2))

        ti.parameters_zero()
        ti.setv(3, 1)
        self.assertEqual(1, ti.getv(3))

        ti.parameters_zero()
        ti.setv(4, 1)
        self.assertEqual(1, ti.getv(4))

        ti.parameters_zero()
        ti.setv(5, 1)
        self.assertEqual(1, ti.getv(5))

        ti.parameters_zero()
        ti.setv(6, 1)
        self.assertEqual(1, ti.getv(6))

        ti.parameters_zero()
        ti.setv(7, 1)
        self.assertEqual(1, ti.getv(7))

        ti.parameters_zero()
        ti.setv(9, 1)
        self.assertEqual(1, ti.getv(9))

        ti.parameters_zero()
        ti.setv(10, 1)
        self.assertEqual(1, ti.getv(10))

        ti.parameters_zero()
        ti.setv(12, 1)
        self.assertEqual(1, ti.getv(12))

        # test raises without all target tensors
        ti.untarget()
        target_tindxs = [1,2]
        ti.setarget(mindxs=None, tindxs=target_tindxs)

        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(8, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test without all target tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        ti.setv(0,1, only_targets=False)
        self.assertEqual(1, ti.getv(0, only_targets=False))
        ti.setv(4,2)
        self.assertEqual(2, ti.getv(4))

        ti.parameters_zero()
        ti.setv(9,1, only_targets=False)
        self.assertEqual(1, ti.getv(9, only_targets=False))
        ti.setv(5,2)
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        ti.setv(11, 1, only_targets=False)
        self.assertEqual(1, ti.getv(11, only_targets=False))
        ti.setv(7,2)
        self.assertEqual(2, ti.getv(7))


class TestIndvdD(unittest.TestCase):
    def test_init(self):
        ti = IndvdD(modules={"1":nn.Linear(2,3), "2":nn.Linear(3,4)})

        self.assertEqual(len(ti.items()), 2)
        self.assertEqual(type(ti.birthtime), datetime)
        self.assertEqual(type(ti.name), str)

    def test_setarget(self):
        til = IndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.setarget(mkeys=["1"])
        for mk, module in til.items():
            for param in module.parameters():
                if mk == "1":
                    self.assertEqual(param.target_torchea, True)
                else:
                    self.assertEqual(param.target_torchea, False)


        til = IndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.setarget(mkeys=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.target_torchea, True)
            else:
                self.assertEqual(param.target_torchea, False)


        til = IndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.setarget()
        for param in til.parameters():
            self.assertEqual(param.target_torchea, True)

    def test_untarget(self):
        tid = IndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        tid.setarget()
        tid.untarget(mkeys=["1"])
        for mk, module in tid.items():
            for param in module.parameters():
                if mk == "1":
                    self.assertEqual(param.target_torchea, False)
                else:
                    self.assertEqual(param.target_torchea, True)


        tid = IndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        tid.setarget()
        tid.untarget(mkeys=None,tindxs=[1])
        for ti, param in enumerate(tid.parameters()):
            if ti == 1:
                self.assertEqual(param.target_torchea, False)
            else:
                self.assertEqual(param.target_torchea, True)


        tid = IndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        tid.setarget()
        tid.untarget()
        for param in tid.parameters():
            self.assertEqual(param.target_torchea, False)

    def test_parameters_zero(self):
        ti = IndvdD(modules={"0":nn.Linear(2,2), "1":nn.Linear(2,3), "2":nn.Linear(3,2)})

        ti.setarget()
        for param in ti.parameters():
            self.assertEqual((param==0).all(), 0)

        ti = IndvdD(modules={"0":nn.Linear(2,2), "1":nn.Linear(2,3), "2":nn.Linear(3,2)})
        ti.untarget(mkeys=["0", "2"])
        for mk, module in ti.items():
            if mk in ["0", "2"]:
                for param in module.parameters():
                    self.assertEqual((param==0).all(), 0)
            else:
                for param in module.parameters():
                    self.assertEqual((param!=0).all(),True)

    def test_count_ws(self):
        ti = IndvdD(modules={"0":nn.Linear(2,2), "1":nn.Linear(2,3), "2":nn.Linear(3,2)})
        ti.setarget()

        self.assertEqual(23, ti.count_ws())

        ti.untarget(mkeys=["0","2"])
        self.assertEqual(9, ti.count_ws())

        self.assertEqual(23, ti.count_ws(only_targets=False))



    def test_getv(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))

        ti.setarget()
        ti.parameters_zero()

        # test raises with all target tensors
        ti.setarget()
        with self.assertRaises(IndexError):
            ti.getv(15)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test with all target tensors
        ti.setarget()
        params = list(ti.parameters())

        ti.parameters_zero()
        params[0].data[0,0] = 1
        self.assertEqual(1, ti.getv(0))

        ti.parameters_zero()
        params[0].data[0,1] = 1
        self.assertEqual(1, ti.getv(1))

        ti.parameters_zero()
        params[0].data[1,1] = 1
        self.assertEqual(1, ti.getv(3))


        ti.parameters_zero()
        params[1].data[0] = 1
        self.assertEqual(1, ti.getv(4))
        ti.parameters_zero()
        params[1].data[1] = 1
        self.assertEqual(1, ti.getv(5))

        ti.parameters_zero()
        params[2].data[0,0] = 1
        self.assertEqual(1, ti.getv(6))

        ti.parameters_zero()
        params[2].data[1,1] = 1
        self.assertEqual(1, ti.getv(9))

        ti.parameters_zero()
        params[2].data[2,0] = 1
        self.assertEqual(1, ti.getv(10))

        ti.parameters_zero()
        params[3].data[0] = 1
        self.assertEqual(1, ti.getv(12))

        ti.parameters_zero()
        params[3].data[1] = 1
        self.assertEqual(1, ti.getv(13))

        ti.parameters_zero()
        params[3].data[2] = 1
        self.assertEqual(1, ti.getv(14))

        # test raises without all target tensors
        target_tindxs = [1,2]
        ti.untarget()
        ti.setarget(mindxs=None, tindxs=target_tindxs)
        with self.assertRaises(IndexError):
            ti.getv(15, only_targets=False)
        with self.assertRaises(IndexError):
            ti.getv(8)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test without all target tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        params[0].data[0,0] = 1
        params[1].data[0] = 2
        self.assertEqual(1, ti.getv(0, only_targets=False))
        self.assertEqual(2, ti.getv(0))

        ti.parameters_zero()
        params[1].data[1] = 1
        params[2].data[1,1] = 2
        self.assertEqual(1, ti.getv(5, only_targets=False))
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        params[2].data[0,1] = 1
        params[2].data[2,1] = 2
        self.assertEqual(1, ti.getv(7, only_targets=False))
        self.assertEqual(2, ti.getv(7))


    def test_setv(self):
        ti = IndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.setarget()
        ti.parameters_zero()

        # test raises with all target tensors
        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test with all target tensors
        ti.parameters_zero()
        ti.setv(0, 1)
        self.assertEqual(1, ti.getv(0))
        
        ti.parameters_zero()
        ti.setv(2, 1)
        self.assertEqual(1, ti.getv(2))

        ti.parameters_zero()
        ti.setv(3, 1)
        self.assertEqual(1, ti.getv(3))

        ti.parameters_zero()
        ti.setv(4, 1)
        self.assertEqual(1, ti.getv(4))

        ti.parameters_zero()
        ti.setv(5, 1)
        self.assertEqual(1, ti.getv(5))

        ti.parameters_zero()
        ti.setv(6, 1)
        self.assertEqual(1, ti.getv(6))

        ti.parameters_zero()
        ti.setv(7, 1)
        self.assertEqual(1, ti.getv(7))

        ti.parameters_zero()
        ti.setv(9, 1)
        self.assertEqual(1, ti.getv(9))

        ti.parameters_zero()
        ti.setv(10, 1)
        self.assertEqual(1, ti.getv(10))

        ti.parameters_zero()
        ti.setv(12, 1)
        self.assertEqual(1, ti.getv(12))

        # test raises without all target tensors
        ti.untarget()
        target_tindxs = [1,2]
        ti.setarget(mindxs=None, tindxs=target_tindxs)

        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(8, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test without all target tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        ti.setv(0,1, only_targets=False)
        self.assertEqual(1, ti.getv(0, only_targets=False))
        ti.setv(4,2)
        self.assertEqual(2, ti.getv(4))

        ti.parameters_zero()
        ti.setv(9,1, only_targets=False)
        self.assertEqual(1, ti.getv(9, only_targets=False))
        ti.setv(5,2)
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        ti.setv(11, 1, only_targets=False)
        self.assertEqual(1, ti.getv(11, only_targets=False))
        ti.setv(7,2)
        self.assertEqual(2, ti.getv(7))


class TestBaseEA(unittest.TestCase):

    def test_init(self):
        ti = IndvdL()

        with self.assertRaises(TypeError):
            bea = BaseEA()
        bea = BaseEA(src_indvd=ti)

    def test_gen_pop(self):
        ti = IndvdL()
        ti.append(nn.Linear(4,2))
        ti.setarget()

        bea = BaseEA(src_indvd=ti)
        bea.gen_pop(npop=10)
        idxs = [id(p) for p in bea]
        self.assertEqual(len(np.unique(idxs)), 10)

        ti = IndvdD()
        ti.update({"1":nn.Linear(4,2)})
        ti.setarget()

        bea = BaseEA(src_indvd=ti)
        bea.gen_pop(npop=10)
        idxs = [id(p) for p in bea]
        self.assertEqual(len(np.unique(idxs)), 10)

    def test_register(self):
        ti = IndvdL()
        bea = BaseEA(src_indvd=ti)

        with self.assertRaises(AttributeError):
            bea.init()
        bea.register("init", uniform)
        self.assertEqual(bea.init(model=ti), None)

    def test_unregister(self):
        ti = IndvdL()
        bea = BaseEA(src_indvd=ti)

        bea.register("init", uniform)
        self.assertEqual(bea.init(model=ti), None)

        bea.unregister("init")
        with self.assertRaises(AttributeError):
            bea.init()
    


if __name__ == "__main__":    
    unittest.main()