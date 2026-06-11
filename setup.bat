@echo off
echo ===================================================
echo   Configuracao do Ambiente - SenseAI (Projeto IA)
echo ===================================================
echo.

echo [1/4] Verificando se o Python 3.11 esta instalado...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python 3.11 nao encontrado no sistema.
    echo Por favor, instale o Python 3.11 do site oficial antes de continuar.
    echo Certifique-se de marcar a opcao "Add Python to PATH" ou usar o instalador oficial do Windows.
    echo.
    pause
    exit /b
)

echo.
echo [2/4] Criando o ambiente virtual (venv) com Python 3.11...
if exist venv (
    echo [AVISO] A pasta 'venv' ja existe. Removendo a versao antiga...
    rmdir /s /q venv
)
py -3.11 -m venv venv

echo.
echo [3/4] Atualizando gerenciadores de pacotes (pip, setuptools, wheel)...
call .\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

echo.
echo [4/4] Instalando todas as dependencias do projeto...
if not exist requirements.txt (
    echo [ERRO] Arquivo requirements.txt nao encontrado na pasta atual!
    pause
    exit /b
)
call .\venv\Scripts\pip.exe install -r requirements.txt

echo.
echo ===================================================
echo   Tudo pronto! O projeto foi configurado com sucesso.
echo ===================================================
echo.
echo Para rodar os scripts ou a API, lembre-se de sempre ativar o ambiente primeiro.
echo Comando para ativar no terminal: .\venv\Scripts\activate
echo.
pause
