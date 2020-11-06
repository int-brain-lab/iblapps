import os
import unittest
from pathlib import Path


class TestTaskQCViewer(unittest.TestCase):
    def setUp(self):
        iblapps_path = Path(__file__).parts[:-2]
        self.task_qc_path = Path(*iblapps_path) / 'task_qc_viewer' / 'task_qc.py'
        self.UUID = '1211f4af-d3e4-4c4e-9d0b-75a0bc2bf1f0'

    def test_build(self):
        os.system(f'python {self.task_qc_path} {self.UUID}')

    def test_build_fail(self):
        os.system(f'python {self.task_qc_path} "BAD_ID"')
