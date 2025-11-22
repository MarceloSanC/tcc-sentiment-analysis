import os
import sys

def find_src_root(start_path="."):
    """Sobe até encontrar 'src' na árvore de diretórios."""
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, "src")):
            return os.path.join(current, "src")
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

def should_ignore(name: str) -> bool:
    """Define quais arquivos/pastas devem ser ignorados."""
    return (
        name.startswith('.') or
        name == "__pycache__" or
        name.endswith('.pyc')
    )

def build_tree_lines(root_dir):
    """
    Retorna uma lista de listas, onde cada sublista representa os componentes de uma linha.
    Ex: ["├──", "domain/", "news.py"] → será combinado com prefixos de indentação.
    """
    output_lines = []
    
    # Pilha: (diretório_atual, caminho_relativo, prefixos_anteriores, é_último?)
    # Mas vamos usar recursão controlada com acumulador para simplicidade.
    
    def recurse(current_path, depth, is_last_stack):
        """
        current_path: caminho absoluto do diretório atual
        depth: nível de profundidade (0 = src/)
        is_last_stack: lista de booleanos indicando se cada nível acima é último
        """
        if depth == 0:
            # Raiz: apenas adiciona "src/"
            output_lines.append(["src/"])
        else:
            # Determina o nome do diretório atual
            dir_name = os.path.basename(current_path) + "/"
            # Monta a linha com base no histórico de "último"
            parts = []
            for i in range(depth - 1):
                if is_last_stack[i]:
                    parts.append("    ")  # espaço vazio
                else:
                    parts.append("│   ")  # linha vertical
            connector = "└── " if is_last_stack[-1] else "├── "
            parts.append(connector)
            parts.append(dir_name)
            output_lines.append(parts)

        # Listar e filtrar conteúdo
        try:
            entries = os.listdir(current_path)
        except OSError:
            return

        entries = [e for e in entries if not should_ignore(e)]
        entries.sort()  # ordenação alfabética

        dirs = [e for e in entries if os.path.isdir(os.path.join(current_path, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(current_path, e))]

        all_items = files + dirs  # arquivos primeiro, depois pastas (opcional; pode inverter)
        total = len(all_items)

        for idx, item in enumerate(all_items):
            is_last = (idx == total - 1)
            full_path = os.path.join(current_path, item)

            # Montar prefixos até este nível
            current_is_last_stack = is_last_stack + [is_last]

            if os.path.isdir(full_path):
                recurse(full_path, depth + 1, current_is_last_stack)
            else:
                # É arquivo: adicionar linha
                parts = []
                for i in range(depth):
                    if is_last_stack[i]:
                        parts.append("    ")
                    else:
                        parts.append("│   ")
                connector = "└── " if is_last else "├── "
                parts.append(connector)
                parts.append(item)
                output_lines.append(parts)

    # Iniciar recursão
    recurse(root_dir, 0, [])
    return output_lines

def main():
    src_path = find_src_root()
    if not src_path or not os.path.isdir(src_path):
        print("❌ Diretório 'src/' não encontrado.", file=sys.stderr)
        sys.exit(1)

    lines_matrix = build_tree_lines(src_path)

    # Imprimir linha por linha, concatenando os componentes
    for line_parts in lines_matrix:
        print("".join(line_parts))

if __name__ == "__main__":
    main()