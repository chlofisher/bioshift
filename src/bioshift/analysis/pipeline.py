from typing import Callable
from pathlib import Path
import yaml

import bioshift.analysis.filters as filters

function_registry: dict[str, Callable] = {
    "normalize": filters.normalize,
    "threshold": filters.threshold,
    "gaussian": filters.gaussian,
    "add": (lambda a, b: a + b),
    "subtract": (lambda a, b: a - b),
    "multiply": (lambda a, b: a * b),
    "divide": (lambda a, b: a / b),
    "difference_of_gaussians": filters.difference_of_gaussians,
}


class Node:
    name: str
    input_keys: list[str]
    output_keys: list[str]
    param_keys: dict[str, str]
    func: Callable

    def __init__(self, name: str, node_dict: dict):
        self.name = name
        self.input_keys = node_dict["inputs"] if "inputs" in node_dict else []
        self.output_keys = node_dict["outputs"] if "outputs" in node_dict else [name]
        self.param_keys = node_dict["params"] if "params" in node_dict else {}
        self.func = function_registry[node_dict["func"]]

    def run(self, context):
        args = [context[key] for key in self.input_keys]
        kwargs = {
            param_key: context[context_key]
            for param_key, context_key in self.param_keys.items()
        }

        result = self.func(*args, **kwargs)

        if isinstance(result, tuple):
            return {
                output_key: val for output_key, val in zip(self.output_keys, result)
            }
        else:
            return {self.output_keys[0]: result}


class Pipeline:
    nodes: dict[str, Node]
    dependencies: dict[str, set[str]]

    def __init__(self, nodes, dependencies):
        self.nodes = nodes
        self.dependencies = dependencies

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path, "r") as file:
            yamldict = yaml.safe_load(file)

        nodes = {}
        dependencies = {}

        for name, node_params in yamldict.items():
            node = Node(name, node_params)
            nodes[name] = node

        for name, node in nodes.items():
            deps = set()

            for input_key in node.input_keys:
                for other_name, other in nodes.items():
                    if input_key in other.output_keys:
                        deps.add(other_name)

            dependencies[name] = deps

        return cls(nodes, dependencies)

    def topological_sort(self) -> list[str]:
        dependencies = {name: deps.copy() for name, deps in self.dependencies.items()}

        in_degree = {name: len(deps) for name, deps in dependencies.items()}

        queue = [name for name in self.nodes.keys() if in_degree[name] == 0]
        sorted = []

        while queue:
            node_name = queue.pop(0)
            sorted.append(node_name)

            for child, deps in dependencies.items():
                if node_name in deps:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

                    deps.remove(node_name)

        if any(dependencies.values()):
            raise ValueError("Pipeline graph is cyclic.")

        return sorted

    def run(self, **kwargs):
        sorted_graph = self.topological_sort()

        context = kwargs
        for node_name in sorted_graph:
            node: Node = self.nodes[node_name]
            result = node.run(context)
            context.update(result)

        return context
