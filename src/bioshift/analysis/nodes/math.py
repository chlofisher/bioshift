
functionregistry = FunctionRegistry()


@functionregistry.register()
def add(a, b):
    return a + b


@functionregistry.register()
def subtract(a, b):
    return a - b


@functionregistry.register()
def multiply(a, b):
    return a * b


@functionregistry.register()
def divide(a, b):
    return a / b
