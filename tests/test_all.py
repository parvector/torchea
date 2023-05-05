import unittest
from test_base import *
from test_cros import *
from test_muts import *



class AllTest(TestBaseIndvdl, TestCros, TestMuts):
    pass

if __name__ == "__main__":
    unittest.main()