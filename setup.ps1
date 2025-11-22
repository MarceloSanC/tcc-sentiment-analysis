# setup.ps1
# Configura o ambiente para rodar o TCC: adiciona make ao PATH e ativa o venv

Write-Host "Certifique-se de executar: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
Write-Host "Configurando ambiente..." -ForegroundColor Cyan

# 1. Adiciona make ao PATH (se não estiver presente)
$makePath = "C:\Program Files (x86)\GnuWin32\bin"
if (Test-Path $makePath) {
    if ($env:PATH -notlike "*$makePath*") {
        $env:PATH += ";$makePath"
        Write-Host "Make adicionado ao PATH (sessão atual)" -ForegroundColor Green
    } else {
        Write-Host "Make já está no PATH" -ForegroundColor Gray
    }
} else {
    Write-Warning "Make não encontrado em $makePath"
    Write-Host "→ Instale com: winget install GnuWin32.Make" -ForegroundColor Yellow
}

# 2. Verifica Python 3.13+
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3\.1[3-9]") {
        Write-Host "Python 3.13+ detectado: $pyVersion" -ForegroundColor Green
    } else {
        throw "Versão incompatível"
    }
} catch {
    Write-Error "Python 3.13+ não encontrado. Instale em: https://python.org"
    exit 1
}

# 3. Ativa o ambiente virtual (se existir)
$venvPath = ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "Ambiente virtual ativado" -ForegroundColor Green
} else {
    Write-Host ".venv não encontrado. Use 'python -m venv .venv' para criar." -ForegroundColor Gray
}

Write-Host "`nPronto! Você pode agora usar:" -ForegroundColor Cyan
Write-Host "   make test     → rodar testes"
Write-Host "   make format   → formatar código"
Write-Host "   make lint     → verificar estilo"