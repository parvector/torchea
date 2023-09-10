import unittest
from torch import nn
from torchea.base import IndvdL, BaseEA
from torchea import desels




class TestDESels(unittest.TestCase):
    def test_softsel(self):
        ti0 = IndvdL(modules=[nn.Linear(2,3)])
        ti0.setarget()
        ti0.fitnes = (0,0,0)
        ti1 = IndvdL(modules=[nn.Linear(2,3)])
        ti1.setarget()
        ti1.fitnes = (0,0,0)

        bea = BaseEA(src_indvd=ti0)
        bea.register("softsel", desels.softsel)
        selti = bea.softsel(ti0, ti1, if_both_eq="first")
        self.assertEqual(selti == ti0, True)

        selti = bea.softsel(ti0, ti1, if_both_eq="second")
        self.assertEqual(selti == ti1, True)

        selti = bea.softsel(ti0, ti1, if_both_eq="both")
        self.assertEqual(selti[0] == ti0, True)
        self.assertEqual(selti[1] == ti1, True)


        ti0.fitnes = (0,0,1)
        ti1.fitnes = (0,0,0)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(selti == ti0, True)

        ti0.fitnes = (0,0,0)
        ti1.fitnes = (0,0,1)
        selti = bea.softsel(ti0, ti1)
        self.assertEqual(selti == ti1, True)

    def test_hardsel(self):
        ti0 = IndvdL(modules=[nn.Linear(2,3)])
        ti0.setarget()
        ti0.fitnes = (0,0,0)
        ti1 = IndvdL(modules=[nn.Linear(2,3)])
        ti1.setarget()
        ti1.fitnes = (0,0,0)

        bea = BaseEA(src_indvd=ti1)
        bea.register("hardsel", desels.hardsel)
        selti = bea.hardsel(ti0, ti1)
        self.assertEqual(selti, False)


        ti0.fitnes = (1,2,3)
        ti1.fitnes = (2,3,4)
        selti = bea.hardsel(ti0, ti1)
        self.assertEqual(ti1 == selti, True)

        ti0.fitnes = (2,3,4)
        ti1.fitnes = (1,2,3)
        selti = bea.hardsel(ti0, ti1)
        self.assertEqual(ti0 == selti, True)

if __name__ == "__main__":    
    unittest.main()