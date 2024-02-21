from collections import OrderedDict
import unittest

from cycleformers.cycles.cycle_utils import CycleSequence

def add_one(x):
    return x + 1

def multiply_two(x):
    return x * 2


class TestCycleSequence(unittest.TestCase):
    def test_valid_args_init(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        self.assertEqual(len(cycle_sequence), 2)
        self.assertEqual(cycle_sequence._methods, OrderedDict({'0': add_one, '1': multiply_two}))

    def test_valid_ordered_dict_init(self):
        methods = OrderedDict({
            'Add One': add_one,
            'Multiply Two': multiply_two
        })
        cycle_sequence = CycleSequence(methods)
        self.assertEqual(len(cycle_sequence), 2)
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'Add One': add_one, 'Multiply Two': multiply_two}))

    def test_empty_args_init(self):
        cycle_sequence = CycleSequence()
        self.assertEqual(len(cycle_sequence), 0)
        self.assertEqual(cycle_sequence._methods, OrderedDict())

    def test__call__(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        result = cycle_sequence(1)
        self.assertEqual(result, 4)

    def test__len__(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        self.assertEqual(len(cycle_sequence), 2)

    def test__add__(self):
        cycle_sequence_1 = CycleSequence(OrderedDict({'func_1': add_one}))
        cycle_sequence_2 = CycleSequence(OrderedDict({'func_2': multiply_two}))

        cycle_sequence = cycle_sequence_1 + cycle_sequence_2
        self.assertEqual(len(cycle_sequence), 2)
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'func_1': add_one, 'func_2': multiply_two}))

    def test__add__name_conflict_idx(self):
        cycle_sequence_1 = CycleSequence(add_one)
        cycle_sequence_2 = CycleSequence(multiply_two)

        cycle_sequence = cycle_sequence_1 + cycle_sequence_2
        self.assertEqual(len(cycle_sequence), 2)
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'0': add_one, '1': multiply_two}))

    def test__add__name_conflict_str(self):
        cycle_sequence_1 = CycleSequence(OrderedDict({'func': add_one}))
        cycle_sequence_2 = CycleSequence(OrderedDict({'func': multiply_two}))

        cycle_sequence = cycle_sequence_1 + cycle_sequence_2
        self.assertEqual(len(cycle_sequence), 2)
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'func': add_one, '1': multiply_two}))

    def test__add__unsupported_type(self):
        cycle_sequence = CycleSequence(add_one)
        with self.assertRaises(ValueError):
            cycle_sequence + 1

    def test__iadd__(self):
        cycle_sequence_1 = CycleSequence(add_one)
        cycle_sequence_2 = CycleSequence(multiply_two)

        cycle_sequence_1 += cycle_sequence_2
        self.assertEqual(len(cycle_sequence_1), 2)
        self.assertDictEqual(cycle_sequence_1._methods, OrderedDict({'0': add_one, '1': multiply_two}))

    def test_get_item_by_idx(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        result = cycle_sequence._get_item_by_idx(cycle_sequence._methods.keys(), 1)
        self.assertEqual(result, '1')

    def test_get_item_by_idx_out_of_range(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        with self.assertRaises(IndexError):
            cycle_sequence._get_item_by_idx(cycle_sequence._methods.keys(), 2)

    def test__getitem__(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        result = cycle_sequence[1]
        self.assertEqual(result, multiply_two)

    # def test__getitem__slice(self):
    #     cycle_sequence = CycleSequence(add_one, multiply_two, add_one)
    #     result = cycle_sequence[0:2]
    #     self.assertEqual(result, multiply_two)
        
    def test__setitem__(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        cycle_sequence[1] = add_one
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'0': add_one, '1': add_one}))

    def test__setitem__out_of_range(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        with self.assertRaises(IndexError):
            cycle_sequence[2] = add_one

    def test__delitem__(self):
        cycle_sequence = CycleSequence(add_one, multiply_two)
        del cycle_sequence[1]
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'0': add_one}))

    def test__delitem__slice(self):
        cycle_sequence = CycleSequence(add_one, multiply_two, add_one)
        del cycle_sequence[0:2]
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'0': add_one}))

    def test_add_method(self):
        cycle_sequence = CycleSequence()
        cycle_sequence.add_method('add_one', add_one)
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'add_one': add_one}))

    def test_add_method_name_conflict(self):
        cycle_sequence = CycleSequence(add_one)
        cycle_sequence.add_method('0', multiply_two)
        self.assertDictEqual(cycle_sequence._methods, OrderedDict({'0': add_one, '1': multiply_two}))