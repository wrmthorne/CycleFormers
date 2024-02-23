# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Tuple

from torch.utils._pytree import SUPPORTED_NODES, LeafSpec, PyTree, TreeSpec, _get_node_type, tree_unflatten


def _is_leaf_or_primitive_container(pytree: PyTree) -> bool:
    """Customized :func:`torch.utils._pytree._is_leaf` to avoid flattening containers of primitives."""
    is_leaf = _get_node_type(pytree) not in SUPPORTED_NODES
    if is_leaf:
        return True

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, _ = flatten_fn(pytree)
    return all(isinstance(child, (int, float, str)) for child in child_pytrees)


def _tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Copy of :func:`torch.utils._pytree.tree_flatten` using our custom leaf function."""
    if _is_leaf_or_primitive_container(pytree):
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    result: List[Any] = []
    children_specs: List["TreeSpec"] = []
    for child in child_pytrees:
        flat, child_spec = _tree_flatten(child)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def _map_and_unflatten(fn: Any, values: List[Any], spec: TreeSpec) -> PyTree:
    """Utility function to apply a function and unflatten it."""
    return tree_unflatten([fn(i) for i in values], spec)