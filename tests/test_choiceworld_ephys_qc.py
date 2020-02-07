import os
import unittest
from pathlib import Path

import ibllib


class TestChoiceWorldEphysQC(unittest.TestCase):
    def setUp(self):
        self.iblapps_path = Path(ibllib.__file__).parent
        self.UUID = "1211f4af-d3e4-4c4e-9d0b-75a0bc2bf1f0"

    def test_build(self):
        os.system(f"ipython {self.iblapps_path}/scripts/choiceworld_ephys_qc.py {self.UUID}")

    def test_build_fail(self):
        os.system(f"ipython {self.iblapps_path}/scripts/choiceworld_ephys_qc.py {"dlsfksdlfkj"}")

    def tearDown(self):
        pass
