@echo off
echo ============================================================
echo INICIANDO EJECUCION SECUENCIAL DE NESTED CV (PAPERMILL)
echo ============================================================
echo.

:: Activar el entorno virtual
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo Entorno virtual activado.
) else (
    echo ADVERTENCIA: No se encontro la carpeta .venv. Se usara el entorno global.
)
echo.

:: Definir comando apuntando a papermill del entorno virtual
set JUPYTER_CMD=.venv\Scripts\python -m papermill

echo [1/4] Ejecutando 01_cnn_lstm.ipynb con Papermill...
%JUPYTER_CMD% selected_notebooks/01_cnn_lstm.ipynb selected_notebooks/01_cnn_lstm.ipynb --cwd selected_notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 01_cnn_lstm.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 01_cnn_lstm.ipynb con exito.
)
echo.

echo [2/4] Ejecutando 02_multimodal_cnn_lstm.ipynb con Papermill...
%JUPYTER_CMD% selected_notebooks/02_multimodal_cnn_lstm.ipynb selected_notebooks/02_multimodal_cnn_lstm.ipynb --cwd selected_notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 02_multimodal_cnn_lstm.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 02_multimodal_cnn_lstm.ipynb con exito.
)
echo.

echo [3/4] Ejecutando 03_mtde_net.ipynb con Papermill...
%JUPYTER_CMD% selected_notebooks/03_mtde_net.ipynb selected_notebooks/03_mtde_net.ipynb --cwd selected_notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 03_mtde_net.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 03_mtde_net.ipynb con exito.
)
echo.

echo [4/4] Ejecutando 04_mtde_net_transfer_learning.ipynb con Papermill...
%JUPYTER_CMD% selected_notebooks/04_mtde_net_transfer_learning.ipynb selected_notebooks/04_mtde_net_transfer_learning.ipynb --cwd selected_notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 04_mtde_net_transfer_learning.ipynb.
) else (
    echo Finalizado 04_mtde_net_transfer_learning.ipynb con exito.
)
echo.

echo ============================================================
echo EJECUCION DE TODOS LOS MODELOS SELECCIONADOS (NCV) FINALIZADA
echo ============================================================
pause
