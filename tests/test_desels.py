import unittest
from torchea.base import BaseIndvdl, BaseEA
from torchea import desels




class TestDESels(unittest.TestCase):
    def test_softsel(self):
        ti0 = BaseIndvdl()
        ti0.eval = (0,0,0)
        ti1 = BaseIndvdl()
        ti1.eval = (0,0,0)

        bea = BaseEA()
        bea.register("softsel", desels.softsel)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(selti.eqid(ti0), True)


        ti0.eval = (0,0,1)
        ti1.eval = (0,0,0)
        bea = BaseEA()
        bea.register("softsel", desels.softsel)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(, ti0)

if __name__ == "__main__":    
    unittest.main()