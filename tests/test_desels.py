import unittest
from torchea.base import BaseIndvd, BaseEA
from torchea import desels




class TestDESels(unittest.TestCase):
    def test_softsel(self):
        ti0 = BaseIndvd()
        ti0.eval = (0,0,0)
        ti1 = BaseIndvd()
        ti1.eval = (0,0,0)

        bea = BaseEA(src_indvd=None)
        bea.register("softsel", desels.softsel)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(selti == ti0, True)

        selti = bea.softsel(ti0, ti1, if_both_eq="second")
        self.assertEqual(selti == ti1, True)

        selti = bea.softsel(ti0, ti1, if_both_eq="both")
        self.assertEqual(selti[0] == ti0, True)
        self.assertEqual(selti[1] == ti1, True)


        ti0.eval = (0,0,1)
        ti1.eval = (0,0,0)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(selti == ti0, True)

        ti0.eval = (0,0,0)
        ti1.eval = (0,0,1)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(selti == ti1, True)

    def test_hardsel(self):
        ti0 = BaseIndvd()
        ti0.eval = (0,0,0)
        ti1 = BaseIndvd()
        ti1.eval = (0,0,0)

        bea = BaseEA(src_indvd=None)
        bea.register("hardsel", desels.hardsel)
        selti = bea.hardsel(ti0, ti1)
        self.assertEqual(selti, False)


        ti0.eval = (1,2,3)
        ti1.eval = (2,3,4)
        selti = bea.hardsel(ti0, ti1)
        self.assertEqual(ti1 == selti, True)

        ti0.eval = (2,3,4)
        ti1.eval = (1,2,3)
        selti = bea.hardsel(ti0, ti1)
        self.assertEqual(ti0 == selti, True)

if __name__ == "__main__":    
    unittest.main()