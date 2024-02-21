from collections import OrderedDict
import unittest

from cycleformers.cycles.cycle_utils import CycleSequence


class TestCycleSequence(unittest.TestCase):
    def setUp(self):
        func_1 = lambda x: x + 1
        func_2 = lambda x: x * 2

    def test_valid_args_init(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        self.assertEqual(len(cycle_sequence), 2)
        self.assertEqual(cycle_sequence._modules, OrderedDict({'0': self.func_1, '1': self.func_2}))

    def test_valid_ordered_dict_init(self):
        methods = OrderedDict({
            'Add One': self.func_1,
            'Multiply Two': self.func_2
        })
        cycle_sequence = CycleSequence(methods)
        self.assertEqual(len(cycle_sequence), 2)
        self.assertEqual(cycle_sequence._modules, OrderedDict({'0': self.func_1, '1': self.func_2}))

    def test_empty_args_init(self):
        cycle_sequence = CycleSequence()
        self.assertEqual(len(cycle_sequence), 0)
        self.assertEqual(cycle_sequence._modules, OrderedDict())

    def test__call__(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        result = cycle_sequence(1)
        self.assertEqual(result, 4)

    def test__len__(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        self.assertEqual(len(cycle_sequence), 2)

    def test__add__(self):
        cycle_sequence_1 = CycleSequence(self.func_1)
        cycle_sequence_2 = CycleSequence(self.func_2)

        cycle_sequence = cycle_sequence_1 + cycle_sequence_2
        self.assertEqual(len(cycle_sequence), 2)
        self.assertEqual(cycle_sequence._modules, OrderedDict({'0': self.func_1, '1': self.func_2}))

    def test_invalid__add__(self):
        cycle_sequence = CycleSequence(self.func_1)
        with self.assertRaises(ValueError):
            cycle_sequence + 1

    def test__iadd__(self):
        cycle_sequence_1 = CycleSequence(self.func_1)
        cycle_sequence_2 = CycleSequence(self.func_2)

        cycle_sequence_1 += cycle_sequence_2
        self.assertEqual(len(cycle_sequence_1), 2)
        self.assertEqual(cycle_sequence_1._modules, OrderedDict({'0': self.func_1, '1': self.func_2}))

    def test_get_item_by_idx(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        result = cycle_sequence._get_item_by_idx(cycle_sequence._modules.keys(), 1)
        self.assertEqual(result, '1')

    def test_get_item_by_idx_out_of_range(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        with self.assertRaises(IndexError):
            cycle_sequence._get_item_by_idx(cycle_sequence._modules.keys(), 2)

    def test__getitem__(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        result = cycle_sequence[1]
        self.assertEqual(result, self.func_2)

    def test__getitem__slice(self):
        cycle_sequence = CycleSequence(self.func_1, self.func_2)
        result = cycle_sequence[0:1]
        self.assertEqual(result, CycleSequence(OrderedDict({'0': self.func_1})))
