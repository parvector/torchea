import unittest
import torch
from torch import nn
from torchea.base import BaseIndvd, BaseIndvdL, BaseIndvdD, BaseEA
from torchea.deinits import uniform
from datetime import datetime
import numpy as np



class TestBaseIndvd(unittest.TestCase):
    def test_init(self):
        ti = BaseIndvd()
        self.assertEqual(type(ti.birthtime), datetime)
        self.assertEqual(type(ti.name), str)

    def test_lt(self):
        ti0 = BaseIndvd()
        ti1 = BaseIndvd()

        ti0.eval = (0,0,0)
        ti1.eval = (1,0,0)
        self.assertEqual(ti0 < ti1, False)

        ti0.eval = (1,2,3)
        ti1.eval = (2,3,4)
        self.assertEqual(ti0 < ti1, True)

    def test_le(self):
        ti0 = BaseIndvd()
        ti1 = BaseIndvd()

        ti0.eval = (0,0,0)
        ti1.eval = (-1,0,0)
        self.assertEqual(ti0 <= ti1, False)

        ti0.eval = (1,2,3)
        ti1.eval = (2,2,3)
        self.assertEqual(ti0 <= ti1, True)

        ti0.eval = (0,0,0)
        ti1.eval = (0,0,0)
        self.assertEqual(ti0 <= ti1, True)

    def test_ne(self):
        ti0 = BaseIndvd()
        ti1 = BaseIndvd()

        ti0.eval = (0,0,0)
        ti1.eval = (1,1,1)
        self.assertEqual(ti0 != ti1, True)

        ti0.eval = (2,3,4)
        ti1.eval = (2,3,4)
        self.assertEqual(ti0 != ti1, False)

    def test_ge(self):
        ti0 = BaseIndvd()
        ti1 = BaseIndvd()

        ti0.eval = (-1,0,0)
        ti1.eval = (0,0,0)
        self.assertEqual(ti0 >= ti1, False)

        ti0.eval = (2,2,3)
        ti1.eval = (1,2,3)
        self.assertEqual(ti0 >= ti1, True)

        ti0.eval = (0,0,0)
        ti1.eval = (0,0,0)
        self.assertEqual(ti0 >= ti1, True)

    def test_gt(self):
        ti0 = BaseIndvd()

        ti1 = BaseIndvd()

        ti0.eval = (0,0,0)
        ti1.eval = (1,0,0)
        self.assertEqual(ti0 > ti1, False)

        ti0.eval = (1,2,3)
        ti1.eval = (2,3,4)
        self.assertEqual(ti1 > ti0, True)

    def test_eq(self):
        ti0 = BaseIndvd()
        ti1 = BaseIndvd()

        ti0.eval = (0,None,0)
        ti1.eval = (0,0,0)
        with self.assertRaises(TypeError):
            ti1 == ti0

        ti0.eval = (0,0,0)
        ti1.eval = (1,0,0)
        self.assertEqual(ti0 == ti1, False)

        ti0.eval = (1,2,3)
        ti1.eval = (1,2,3)
        self.assertEqual(ti1 == ti0, True)

class TestBaseIndvdL(unittest.TestCase):
    def test_init(self):
        ti = BaseIndvdL(modules=[nn.Linear(2,3), nn.Linear(3,4)])
        self.assertEqual(len(ti), 2)
        self.assertEqual(type(ti.birthtime), datetime)
        self.assertEqual(type(ti.name), str)

    def test_freeze(self):
        til = BaseIndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.freeze(mindxs=[1])
        for mi, module in enumerate(til):
            for param in module.parameters():
                if mi == 1:
                    self.assertEqual(param.requires_grad, False)
                else:
                    self.assertEqual(param.requires_grad, True)


        til = BaseIndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.freeze(mindxs=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.requires_grad, False)
            else:
                self.assertEqual(param.requires_grad, True)


        til = BaseIndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.freeze()
        for param in til.parameters():
            self.assertEqual(param.requires_grad, False)

    def test_unfreeze(self):
        til = BaseIndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.freeze()
        til.unfreeze(mindxs=[1])
        for mi, module in enumerate(til):
            for param in module.parameters():
                if mi == 1:
                    self.assertEqual(param.requires_grad, True)
                else:
                    self.assertEqual(param.requires_grad, False)


        til = BaseIndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.freeze()
        til.unfreeze(mindxs=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.requires_grad, True)
            else:
                self.assertEqual(param.requires_grad, False)


        til = BaseIndvdL()
        til.append(nn.Linear(2,2))
        til.append(nn.Linear(2,3))
        til.append(nn.Linear(3,2))

        til.freeze()
        til.unfreeze()
        for param in til.parameters():
            self.assertEqual(param.requires_grad, True)

    def test_count_ws(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,2))
        self.assertEqual(23, ti.count_ws(only_freeze=False))
        for param in ti[1].parameters():
            param.requires_grad = False
        self.assertEqual(9, ti.count_ws())


    def test_getv(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))

        ti.freeze()
        ti.parameters_zero()

        # test raises with all freeze tensors
        ti.freeze()
        with self.assertRaises(IndexError):
            ti.getv(15)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test with all freeze tensors
        ti.freeze()
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

        # test raises without all freeze tensors
        freeze_tindxs = [1,2]
        ti.unfreeze()
        ti.freeze(mindxs=None, tindxs=freeze_tindxs)
        with self.assertRaises(IndexError):
            ti.getv(15, only_freeze=False)
        with self.assertRaises(IndexError):
            ti.getv(8)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test without all freeze tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        params[0].data[0,0] = 1
        params[1].data[0] = 2
        self.assertEqual(1, ti.getv(0, only_freeze=False))
        self.assertEqual(2, ti.getv(0))

        ti.parameters_zero()
        params[1].data[1] = 1
        params[2].data[1,1] = 2
        self.assertEqual(1, ti.getv(5, only_freeze=False))
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        params[2].data[0,1] = 1
        params[2].data[2,1] = 2
        self.assertEqual(1, ti.getv(7, only_freeze=False))
        self.assertEqual(2, ti.getv(7))


    def test_setv(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.freeze()
        ti.parameters_zero()

        # test raises with all freeze tensors
        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test with all freeze tensors
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

        # test raises without all freeze tensors
        ti.unfreeze()
        freeze_tindxs = [1,2]
        ti.freeze(mindxs=None, tindxs=freeze_tindxs)

        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(8, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test without all freeze tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        ti.setv(0,1, only_freeze=False)
        self.assertEqual(1, ti.getv(0, only_freeze=False))
        ti.setv(4,2)
        self.assertEqual(2, ti.getv(4))

        ti.parameters_zero()
        ti.setv(9,1, only_freeze=False)
        self.assertEqual(1, ti.getv(9, only_freeze=False))
        ti.setv(5,2)
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        ti.setv(11, 1, only_freeze=False)
        self.assertEqual(1, ti.getv(11, only_freeze=False))
        ti.setv(7,2)
        self.assertEqual(2, ti.getv(7))


class TestBaseIndvdD(unittest.TestCase):
    def test_init(self):
        ti = BaseIndvdD(modules={"1":nn.Linear(2,3), "2":nn.Linear(3,4)})

        self.assertEqual(len(ti.items()), 2)
        self.assertEqual(type(ti.birthtime), datetime)
        self.assertEqual(type(ti.name), str)

    def test_freeze(self):
        til = BaseIndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.freeze(mkeys=["1"])
        for mk, module in til.items():
            for param in module.parameters():
                if mk == "1":
                    self.assertEqual(param.requires_grad, False)
                else:
                    self.assertEqual(param.requires_grad, True)


        til = BaseIndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.freeze(mkeys=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.requires_grad, False)
            else:
                self.assertEqual(param.requires_grad, True)


        til = BaseIndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.freeze()
        for param in til.parameters():
            self.assertEqual(param.requires_grad, False)

    def test_unfreeze(self):
        til = BaseIndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.freeze()
        til.unfreeze(mkeys=["1"])
        for mk, module in til.items():
            for param in module.parameters():
                if mk == "1":
                    self.assertEqual(param.requires_grad, True)
                else:
                    self.assertEqual(param.requires_grad, False)


        til = BaseIndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.freeze()
        til.unfreeze(mkeys=None,tindxs=[1])
        for ti, param in enumerate(til.parameters()):
            if ti == 1:
                self.assertEqual(param.requires_grad, True)
            else:
                self.assertEqual(param.requires_grad, False)


        til = BaseIndvdD(modules={"1":nn.Linear(2,2), 
                                  "2":nn.Linear(2,3), 
                                  "3":nn.Linear(3,2)})

        til.freeze()
        til.unfreeze()
        for param in til.parameters():
            self.assertEqual(param.requires_grad, True)

    def test_count_ws(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,2))
        self.assertEqual(23, ti.count_ws(only_freeze=False))
        for param in ti[1].parameters():
            param.requires_grad = False
        self.assertEqual(9, ti.count_ws())


    def test_getv(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))

        ti.freeze()
        ti.parameters_zero()

        # test raises with all freeze tensors
        ti.freeze()
        with self.assertRaises(IndexError):
            ti.getv(15)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test with all freeze tensors
        ti.freeze()
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

        # test raises without all freeze tensors
        freeze_tindxs = [1,2]
        ti.unfreeze()
        ti.freeze(mindxs=None, tindxs=freeze_tindxs)
        with self.assertRaises(IndexError):
            ti.getv(15, only_freeze=False)
        with self.assertRaises(IndexError):
            ti.getv(8)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        # test without all freeze tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        params[0].data[0,0] = 1
        params[1].data[0] = 2
        self.assertEqual(1, ti.getv(0, only_freeze=False))
        self.assertEqual(2, ti.getv(0))

        ti.parameters_zero()
        params[1].data[1] = 1
        params[2].data[1,1] = 2
        self.assertEqual(1, ti.getv(5, only_freeze=False))
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        params[2].data[0,1] = 1
        params[2].data[2,1] = 2
        self.assertEqual(1, ti.getv(7, only_freeze=False))
        self.assertEqual(2, ti.getv(7))


    def test_setv(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.freeze()
        ti.parameters_zero()

        # test raises with all freeze tensors
        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test with all freeze tensors
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

        # test raises without all freeze tensors
        ti.unfreeze()
        freeze_tindxs = [1,2]
        ti.freeze(mindxs=None, tindxs=freeze_tindxs)

        with self.assertRaises(IndexError):
            ti.setv(15, 1)
        with self.assertRaises(IndexError):
            ti.setv(8, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        # test without all freeze tensors
        params = list(ti.parameters())
        ti.parameters_zero()
        ti.setv(0,1, only_freeze=False)
        self.assertEqual(1, ti.getv(0, only_freeze=False))
        ti.setv(4,2)
        self.assertEqual(2, ti.getv(4))

        ti.parameters_zero()
        ti.setv(9,1, only_freeze=False)
        self.assertEqual(1, ti.getv(9, only_freeze=False))
        ti.setv(5,2)
        self.assertEqual(2, ti.getv(5))

        ti.parameters_zero()
        ti.setv(11, 1, only_freeze=False)
        self.assertEqual(1, ti.getv(11, only_freeze=False))
        ti.setv(7,2)
        self.assertEqual(2, ti.getv(7))


class TestBaseEA(unittest.TestCase):

    def test_init(self):
        ti = BaseIndvdL()

        with self.assertRaises(TypeError):
            bea = BaseEA()
        bea = BaseEA(src_indvd=ti)

    def test_gen_pop(self):
        ti = BaseIndvdL()
        ti.append(nn.Linear(4,2))
        bea = BaseEA(src_indvd=ti)
        bea.gen_pop(npop=10)
        idxs = [id(p) for p in bea]
        self.assertEqual(len(np.unique(idxs)), 10)

        ti = BaseIndvdD()
        ti.update({"1":nn.Linear(4,2)})
        bea = BaseEA(src_indvd=ti)
        bea.gen_pop(npop=10)
        idxs = [id(p) for p in bea]
        self.assertEqual(len(np.unique(idxs)), 10)

    def test_register(self):
        ti = BaseIndvdL()
        bea = BaseEA(src_indvd=ti)

        with self.assertRaises(AttributeError):
            bea.init()
        bea.register("init", uniform)
        self.assertEqual(bea.init(model=ti), None)

    def test_unregister(self):
        ti = BaseIndvdL()
        bea = BaseEA(src_indvd=ti)

        bea.register("init", uniform)
        self.assertEqual(bea.init(model=ti), None)

        bea.unregister("init")
        with self.assertRaises(AttributeError):
            bea.init()
    


if __name__ == "__main__":    
    unittest.main()