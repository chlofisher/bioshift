from typing import Callable


class NodeRegistry:
    functions: dict[str, Callable]

    def __init__(self):
        self.functions = {}

    def get_function(self, key: str) -> Callable:
        return self.functions[key]

    def register(self, name=None):
        def decorator(func):
            key = name or func.__name__
            if key in self.functions:
                raise ValueError(f"Duplicate entry {key} in function registry.")

            self.functions[key] = func
            return func

        return decorator
