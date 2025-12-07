import os
import sys

# Pastas que devem ser ignoradas completamente
HEAVY_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build"
}

# Itens ocultos e arquivos desnecessários
IGNORE_ITEMS = {
    ".git",
    ".idea",
    ".DS_Store"
}

def find_project_root(start_path="."):
    """
    Detecta a raiz do projeto subindo diretórios até encontrar marcadores conhecidos.
    """
    current = os.path.abspath(start_path)

    while True:
        markers = [".git", "pyproject.toml", "package.json"]
        if any(os.path.exists(os.path.join(current, m)) for m in markers):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            return current
        current = parent


def should_ignore(name):
    """Define se um arquivo ou diretório deve ser ignorado."""
    if name in HEAVY_DIRS:
        return True
    if name in IGNORE_ITEMS:
        return True
    if name.startswith(".") and name not in {".git"}:
        return True
    return False


def build_tree(root_dir):
    """
    Constrói a estrutura hierárquica exatamente como o tree original.
    """
    output = []

    def recurse(path, prefix=""):
        items = sorted(
            [i for i in os.listdir(path) if not should_ignore(i)],
            key=lambda x: (os.path.isfile(os.path.join(path, x)), x)
        )
        total = len(items)

        for index, item in enumerate(items):
            full_path = os.path.join(path, item)
            is_last = index == total - 1

            connector = "└── " if is_last else "├── "
            output.append(prefix + connector + item + ("/" if os.path.isdir(full_path) else ""))

            if os.path.isdir(full_path):
                new_prefix = prefix + ("    " if is_last else "│   ")
                recurse(full_path, new_prefix)

    project_name = os.path.basename(root_dir.rstrip("/"))
    output.append(project_name + "/")
    recurse(root_dir)
    return output


def main():
    root = find_project_root()

    if not root:
        print("❌ Não foi possível localizar a raiz do projeto.", file=sys.stderr)
        sys.exit(1)

    lines = build_tree(root)

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
