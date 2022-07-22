import unittest
from torch import nn
from torchea.base import BaseIndvdl



class TestBase(unittest.TestCase):

    def test_BaseIndvdl(self):
        # test init
        bi  = BaseIndvdl()
        self.assertIsInstance(bi, BaseIndvdl)

        # test append
        bi.append(nn.Linear(1,2))
        isrequires_grad = list(bi.parameters())[0].requires_grad
        self.assertFalse(isrequires_grad)
        
        # test insert
        bi.insert(0,nn.Linear(2,3))
        shape = tuple(bi[0].weight.shape)
        self.assertEqual(shape, (3,2))
        isrequires_grad = list(bi[0].parameters())[0].requires_grad
        self.assertFalse(isrequires_grad)

        # test expend
        bi.extend(nn.ModuleList([nn.Linear(3,4), nn.Linear(4,5)]))
        self.assertEqual(len(bi), 4)
        for i in range(2,4):
            isrequires_grad = list(bi[i].parameters())[0].requires_grad
            self.assertFalse(isrequires_grad)

        # test pop
        bi.pop(0)
        self.assertEqual(len(bi), 3)


if __name__ == "__main__":    
    unittest.main()