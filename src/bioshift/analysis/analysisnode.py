from typing import Callable
from functools import wraps


class AnalysisNode:
    func: Callable

    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.params = kwargs

    def run(self, input):
        return self.func(input, **self.params)


def analysis_node(func: Callable):
    @wraps(func)
    def wrapper(**kwargs) -> AnalysisNode:
        return AnalysisNode(func, **kwargs)

    return wrapper
