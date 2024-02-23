import gc
import unittest



class CycleModelTester:

    def tearDown(self):
        gc.collect()


    