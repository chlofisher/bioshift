import importlib
import sys
import pdoc

import pdoc.doc
import pdoc.docstrings


def main():
    mod_name = sys.argv[1]
    title = "Public API"
    mod = importlib.import_module(mod_name)
    mod_doc = pdoc.doc.Module(mod)

    print(f"---\ntitle: {title}\n---\n")
    print("""import { Card } from '@astrojs/starlight/components';\n""")
    # crawl_recursive(mod_doc, depth=0)
    crawl(mod_doc)


def crawl(doc):
    stack = [doc]
    depth_stack = [0]

    while len(stack) > 0:
        curr_doc = stack.pop(-1)
        curr_depth = depth_stack.pop(-1)

        if not is_private(curr_doc):
            render_markdown(curr_doc, curr_depth)

        if not isinstance(curr_doc, pdoc.doc.Namespace):
            continue

        members = list(curr_doc.own_members)
        for member in members[::-1]:
            stack.append(member)
            depth_stack.append(curr_depth + 1)


def crawl_recursive(doc, depth):
    if is_private(doc):
        return

    if depth > 0:
        print(f"""<Card title="{doc.name}">""")

    render_markdown(doc, depth=1)

    if isinstance(doc, pdoc.doc.Namespace):
        members = list(doc.own_members)
        for member in members:
            crawl_recursive(member, depth=depth + 1)

    if depth > 0:
        print("</Card>")


def render_markdown(doc, depth):
    # if depth == 1:
    #     print("\n---")

    if depth < 1:
        return

    header_level = 2 * depth
    print(f"""{"#" * header_level} {doc.name}""")

    if isinstance(doc, pdoc.doc.Function):
        print("```python")
        for decorator in doc.decorators:
            print(f"{decorator}")

        print(f"{doc.funcdef} {doc.name}{doc.signature}:")
        print("```")
    elif isinstance(doc, pdoc.doc.Class):
        print("```python")
        definition = f"class {doc.name}"

        if len(doc.bases) > 0:
            definition += f"""({", ".join([base[2] for base in doc.bases])})"""

        definition += ":"

        print(definition)
        print("```")
    elif isinstance(doc, pdoc.doc.Variable):
        print("```python")
        definition = f"{doc.name}{doc.annotation_str}"
        default = doc.default_value_str
        if default:
            definition += f" = {default}"
        print(definition)

        print("```")

    formatted_docstring = pdoc.docstrings.google(doc.docstring)
    print(formatted_docstring)


def is_private(doc):
    return doc.name[0] == "_"


def is_dunder(doc):
    return doc.name[:2] == "__"


if __name__ == "__main__":
    main()
