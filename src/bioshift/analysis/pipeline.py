import numpy as np
from typing import Protocol, Callable, Any
from pathlib import Path
import yaml

from bioshift.analysis.node import Node

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
            NodeType = node_registry[node_params["type"]]
            node = NodeType(name, node_params)
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
        dependencies = {name: deps.copy()
                        for name, deps in self.dependencies.items()}

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
