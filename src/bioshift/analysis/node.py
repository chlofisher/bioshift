from typing import Protocol, Callable


class Node(Protocol):


    def __init__(self, name: str, nod_dict: dict): ...

    def run(self, context): ...

class NodeInstance:
    name: str
    input_keys: list[str]
    output_keys: list[str]
    param_keys: dict[str, str]
    ...


class FunctionNode:
    func: Callable

    def __init__(self, name: str, node_dict: dict):
        self.name = name
        self.input_keys = node_dict["inputs"] if "inputs" in node_dict else []
        self.output_keys = node_dict["outputs"] if "outputs" in node_dict else [
            name]
        self.param_keys = node_dict["params"] if "params" in node_dict else {}
        self.func = functionregistry.get_function(node_dict["func"])

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


class ConstantNode:
    value: Any

    def __init__(self, name: str, node_dict: dict):
        self.name = name
        self.input_keys = []
        self.output_keys = [name]
        self.param_keys = {}

        value = node_dict["value"]
        if type(value) == list:
            self.value = np.array(value)
        else:
            self.value = node_dict["value"]

    def run(self, context):
        return {self.output_keys[0]: self.value}
