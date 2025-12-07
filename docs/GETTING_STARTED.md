# 🚀 Primeiros Passos

Este guia explica como configurar e executar o projeto localmente (Windows).

## Pré-requisitos

- Windows 10/11
- Python 3.13+
- Git (opcional)

## Configuração

1. Clone o repositório (ou baixe o ZIP):
```
git clone https://github.com/MarceloSanC/tcc-sentiment-analysis.git
cd tcc-sentiment-analysis
```

2. Instale o `make` (via winget):
```
winget install GnuWin32.Make
```

3. Configure o ambiente:
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
```

> O script `setup.ps1`:
> - Adiciona `make` ao PATH
> - Verifica Python 3.13+
> - Ativa o ambiente virtual (se `.venv` existir)

4. Crie o ambiente virtual (se ainda não existir):
```
python -m venv .venv
.\setup.ps1
```

5. Instale as dependências:
```
make install
```

Pronto! O projeto está configurado.