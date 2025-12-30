# Standard libraries
import argparse
import json
import re
from pathlib import Path
from typing import Any


def markdown_to_notebook(markdown_content: str) -> dict[str, Any]:
    """
    Convierte contenido Markdown a formato Jupyter Notebook.
    Los bloques ```python``` se convierten en celdas de código.
    Los títulos (#, ##, etc.) se separan en celdas individuales.
    """

    notebook: dict[str, Any] = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Dividir por líneas para procesar línea por línea
    lines = markdown_content.split("\n")

    current_block = []
    current_type = None
    in_code_block = False
    code_content = []

    for line in lines:
        # Detectar inicio de bloque de código Python
        if line.strip() == "```python":
            # Guardar bloque anterior si existe
            if current_block:
                content = "\n".join(current_block).strip()
                if content:
                    notebook["cells"].append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": content.split("\n"),
                        }
                    )
                current_block = []

            in_code_block = True
            code_content = []
            continue

        # Detectar fin de bloque de código
        if in_code_block and line.strip() == "```":
            code_text = "\n".join(code_content).strip()
            if code_text:
                notebook["cells"].append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": code_text.split("\n"),
                    }
                )
            in_code_block = False
            code_content = []
            continue

        # Si estamos dentro de un bloque de código, agregar línea
        if in_code_block:
            code_content.append(line)
            continue

        # Detectar títulos
        header_match = re.match(r"^(#{1,6}\s+.+)$", line)
        if header_match:
            # Guardar bloque anterior si existe
            if current_block:
                content = "\n".join(current_block).strip()
                if content:
                    notebook["cells"].append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": content.split("\n"),
                        }
                    )
                current_block = []

            # Agregar título como celda separada
            notebook["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line],
                }
            )
            continue

        # Línea normal de markdown
        current_block.append(line)

    # Guardar último bloque si existe
    if current_block:
        content = "\n".join(current_block).strip()
        if content:
            notebook["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": content.split("\n"),
                }
            )

    return notebook


def convert_file(input_path: str, output_path: str | None = None) -> Path:
    """
    Convierte un archivo Markdown a Jupyter Notebook.

    Args:
        input_path: Ruta al archivo .md
        output_path: Ruta de salida para el .ipynb (por defecto: mismo nombre con .ipynb)

    Returns:
        Path del archivo generado
    """

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"El archivo {input_file} no existe")

    with open(input_file, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    notebook = markdown_to_notebook(markdown_content)

    output_file = Path(output_path) if output_path else input_file.with_suffix(".ipynb")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"✓ Notebook creado: {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte archivos Markdown a Jupyter Notebooks"
    )
    parser.add_argument("input", type=str, help="Archivo Markdown de entrada (.md)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Archivo de salida (.ipynb). Por defecto: mismo nombre que entrada con extensión .ipynb",
    )

    args = parser.parse_args()

    try:
        convert_file(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
