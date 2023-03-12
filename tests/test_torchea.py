import unittest, torch
from torch import nn
from torchea.base import BaseIndvdl
from torchea import croses, muts



class TestBase(unittest.TestCase):

    def test_init(self):
        ti = BaseIndvdl()
        self.assertIsInstance(ti, BaseIndvdl)

    def test_insert(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(1,2))
        ti.insert(0,nn.Linear(2,3))
        shape = tuple(ti[0].weight.shape)
        self.assertEqual(shape, (3,2))
        isrequires_grad = list(ti[0].parameters())[0].requires_grad
        self.assertFalse(isrequires_grad)
        
    def test_append(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(1,2))
        isrequires_grad = list(ti.parameters())[0].requires_grad
        self.assertFalse(isrequires_grad)

    def test_expend(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(1,2))
        ti.insert(0,nn.Linear(2,3))
        ti.extend( [nn.Linear(3,4), nn.Linear(4,5)] )
        self.assertEqual(len(ti), 4)
        for i in range(2,4):
            isrequires_grad = list(ti[i].parameters())[0].requires_grad
            self.assertFalse(isrequires_grad)

    def test_pop(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(1,2))
        ti.insert(0,nn.Linear(2,3))
        ti.extend(nn.ModuleList([nn.Linear(3,4), nn.Linear(4,5)]))
        ti.pop(0)
        self.assertEqual(len(ti), 3)

    def test_parameters_zero(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(1,2))
        ti.parameters_zero()
        for params in ti.parameters():
            self.assertTrue((params.data == 0).all().item())

    def test_freeze_module(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,2))
        for param in ti.parameters():
            self.assertEqual(param.requires_grad, False)


    def test_count_params(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,2))
        ti_count_param = 23
        self.assertEqual(ti_count_param, ti.count_params())

    def test_getv(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,2))
        params = list(ti.parameters())

        with self.assertRaises(IndexError):
            ti.getv(100)
        with self.assertRaises(IndexError):
            ti.getv(-1)

        ti.parameters_zero()
        params[0].data[0,0] = 1
        self.assertEqual(1, ti.getv(0))

        ti.parameters_zero()
        params[1].data[0] = 1
        self.assertEqual(1, ti.getv(4))

        ti.parameters_zero()
        params[1].data[1] = 1
        self.assertEqual(1, ti.getv(5))

        ti.parameters_zero()
        params[2].data[2,1] = 1
        self.assertEqual(1, ti.getv(11))


    def test_setv(self):
        ti = BaseIndvdl()
        ti.append(nn.Linear(2,2))
        ti.append(nn.Linear(2,3))
        ti.append(nn.Linear(3,2))
        ti.parameters_zero()

        with self.assertRaises(IndexError):
            ti.setv(100, 1)
        with self.assertRaises(IndexError):
            ti.setv(-1, 1)

        ti.setv(0, 1)
        self.assertEqual(1, ti.getv(0))
        ti.parameters_zero()

        ti.setv(4, 1)
        self.assertEqual(1, ti.getv(4))
        ti.parameters_zero()

        ti.setv(5, 1)
        self.assertEqual(1, ti.getv(5))
        ti.parameters_zero()

        ti.setv(11, 1)
        self.assertEqual(1, ti.getv(11))
        ti.parameters_zero()

class TestCroses(unittest.TestCase):
    def test_crosDE(self):
        ti0 = BaseIndvdl()
        ti0.append(nn.Linear(2,2))

        ti1 = BaseIndvdl()
        ti1.append(nn.Linear(2,2))
        croses.crosDE(ti0, ti1)
        self.assertEqual()

class TestMuts(unittest.TestCase):

    def test_mutDE(self):
        ti1 = BaseIndvdl()
        ti1.append(nn.Linear(2,2))
        ti1.parameters_zero()
        for params in ti1.parameters():
            params+=2

        ti2 = BaseIndvdl()
        ti2.append(nn.Linear(2,2))
        ti2.parameters_zero()
        for params in ti2.parameters():
            params+=2

        ti3 = BaseIndvdl()
        ti3.append(nn.Linear(2,2))
        ti3.parameters_zero()
        for params in ti3.parameters():
            params+=1

        #test mutDE1
        mut_model = muts.mutDE(ti1, ti2, ti3, 2)   
        for mutv in mut_model.parameters():
            self.assertTrue((mutv==4).all().item())

if __name__ == "__main__":    
    unittest.main()