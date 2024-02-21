from collections import OrderedDict
from itertools import islice
import operator
from typing import Callable, Dict, Iterator, List, overload, Union

# Adaptation of torch.nn.Sequential https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py#L43
class CycleSequence:

    _methods: Dict[str, Callable] = OrderedDict()

    @overload
    def __init__(self, *args: Callable) -> None:
        ...

    @overload
    def __init__(self, arg: OrderedDict[str, Callable]) -> None:
        ...

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, method in args[0].items():
                self.add_method(key, method)
        else:
            for idx, method in enumerate(args):
                self.add_method(str(idx), method)

    def __call__(self, input):
        for method in self._methods.values():
            input = method(input)
        return input
    
    def __len__(self):
        return len(self._methods)
    
    def __add__(self, other) -> 'CycleSequence':
        if isinstance(other, CycleSequence):
            for name, method in other._methods.items():
                self.add_method(name, method)
        else:
            raise ValueError(f'Unsupported type {type(other)}')
        
        return self
    
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
        key: str = self._get_item_by_idx(self._methods.keys(), idx)
        return setattr(self, key, method)
    
    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._methods.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._methods.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._methods))]
        self._methods = OrderedDict(list(zip(str_indices, self._methods.values())))

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys
    
    def __iter__(self) -> Iterator[Callable]:
        return iter(self._methods.values())
    
    def __repr__(self) -> str:
        text = self.__class__.__name__ + '\n'
        for i, (name, method) in enumerate(self._methods.items()):
            text += f'\n\t({i}): {name} -> {method}'
        return text
    
    def add_method(self, name: str, method: Callable) -> None:
        self._methods[name] = method

    def insert(self, index: int, method: Callable) -> 'CycleSequence':
        if not isinstance(method, Callable):
            raise AssertionError(
                f'method should be of type: {Callable}')
        n = len(self._methods)
        if not (-n <= index <= n):
            raise IndexError(
                f'Index out of range: {index}')
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._methods[str(i)] = self._methods[str(i - 1)]
        self._methods[str(index)] = method
        return self
    
    def append(self, other: Union[Callable, OrderedDict]):
        if isinstance(other, OrderedDict):
            for name, method in method.items():
                self.add_method(name, method)
        else:
            self.add_method(str(len(self._methods)), other)
    
    def extend(self, *args: Union[Callable, OrderedDict]):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, method in args[0].items():
                self.add_method(key, method)
        else:
            start_idx = len(self._methods)
            for idx, method in enumerate(args):
                self.add_method(str(start_idx + idx), method)
    
    