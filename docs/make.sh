#!/usr/bin/env bash
DOCS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

uv run $DOCS_DIR/markdowndocs.py bioshift > $DOCS_DIR/src/content/docs/reference/api.mdx
