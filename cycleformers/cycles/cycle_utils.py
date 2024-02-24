from collections import OrderedDict
from itertools import islice
import operator
from typing import Callable, Dict, Iterator, Optional, overload, Union

# Adaptation of torch.nn.Sequential https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py#L43
class CycleSequence:

    _methods: Dict[str, Callable]

    @overload
    def __init__(self, *args: Callable) -> None:
        ...

    @overload
    def __init__(self, arg: OrderedDict[str, Callable]) -> None:
        ...

    def __init__(self, *args):
        super().__setattr__('_methods', OrderedDict())

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, method in args[0].items():
                self.add_method(key, method)
        else:
            for idx, method in enumerate(args):
                self.add_method(str(idx), method)

    def __call__(self, **inputs):
        for method in self._methods.values():
            inputs = method(**inputs)
        return inputs
    
    def __len__(self):
        return len(self._methods)
    
    def __add__(self, other) -> 'CycleSequence':
        if isinstance(other, CycleSequence):
            new_sequence = CycleSequence(self._methods)

            for name, method in other._methods.items():
                new_sequence.add_method(name, method)

            return new_sequence
        else:
            raise ValueError(f'Unsupported type {type(other)}')
    
    def __iadd__(self, other) -> 'CycleSequence':
        return self.__add__(other)
    
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f'index {idx} is out of range')
        idx %= size
        return next(islice(iterator, idx, None))
    
    def __getitem__(self, idx: Union[int, str]) -> Callable:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._methods.items())[idx]))
        else:
            return self._get_item_by_idx(self._methods.values(), idx)
        
    def __setitem__(self, idx: int, method: Callable) -> None:
        key = self._get_item_by_idx(self._methods.keys(), idx)
        self._methods[key] = method
    
    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._methods.keys())[idx]:
                del self._methods[key]
        else:
            key = self._get_item_by_idx(self._methods.keys(), idx)
            del self._methods[key]
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._methods))]
        self._methods = OrderedDict(list(zip(str_indices, self._methods.values())))
    
    def __iter__(self) -> Iterator[Callable]:
        return iter(self._methods.values())
    
    def __repr__(self) -> str:
        text = self.__class__.__name__ + '\n'
        for i, (name, method) in enumerate(self._methods.items()):
            text += f'\n\t({i}): {name} -> {method}'
        return text
    
    def add_method(self, name: str, method: Callable) -> None:
        if name in self._methods:
            name = str(len(self._methods))

        self._methods[name] = method
    
    @overload
    def append(self, other: Callable) -> None:
        ...

    @overload
    def append(self, other: OrderedDict[str, Callable]) -> None:
        ...
    
    def append(self, other):
        self.add_method(str(len(self._methods)), other)
    
    @overload
    def extend(self, *args: Callable) -> None:
        ...

    @overload
    def extend(self, args: OrderedDict[str, Callable]) -> None:
        ...

    def extend(self, sequence):
        if isinstance(sequence, CycleSequence):
            for method in sequence._methods.values():
                self.append(method)
        elif isinstance(sequence, OrderedDict):
            for name, method in sequence.items():
                self.add_method(name, method)
        else:
            self.append(method)