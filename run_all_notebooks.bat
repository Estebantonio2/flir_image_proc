@echo off
echo ============================================================
echo INICIANDO EJECUCION SECUENCIAL DE ENTRENAMIENTOS (PAPERMILL)
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

echo [1/6] Ejecutando 01_DSTFS_training.ipynb con Papermill...
%JUPYTER_CMD% notebooks/01_DSTFS_training.ipynb notebooks/01_DSTFS_training.ipynb --cwd notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 01_DSTFS_training.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 01_DSTFS_training.ipynb con exito.
)
echo.

echo [2/6] Ejecutando 02_MTDE_Net_training.ipynb con Papermill...
%JUPYTER_CMD% notebooks/02_MTDE_Net_training.ipynb notebooks/02_MTDE_Net_training.ipynb --cwd notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 02_MTDE_Net_training.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 02_MTDE_Net_training.ipynb con exito.
)
echo.

echo [3/6] Ejecutando 03_CNN_LSTM_training.ipynb con Papermill...
%JUPYTER_CMD% notebooks/03_CNN_LSTM_training.ipynb notebooks/03_CNN_LSTM_training.ipynb --cwd notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 03_CNN_LSTM_training.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 03_CNN_LSTM_training.ipynb con exito.
)
echo.

echo [4/6] Ejecutando 04_1D_CNN_training.ipynb con Papermill...
%JUPYTER_CMD% notebooks/04_1D_CNN_training.ipynb notebooks/04_1D_CNN_training.ipynb --cwd notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 04_1D_CNN_training.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 04_1D_CNN_training.ipynb con exito.
)
echo.

echo [5/6] Ejecutando 05_MTDE_Net_transfer_learning.ipynb con Papermill...
%JUPYTER_CMD% notebooks/05_MTDE_Net_transfer_learning.ipynb notebooks/05_MTDE_Net_transfer_learning.ipynb --cwd notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 05_MTDE_Net_transfer_learning.ipynb. Continuando con el siguiente...
) else (
    echo Finalizado 05_MTDE_Net_transfer_learning.ipynb con exito.
)
echo.

echo [6/6] Ejecutando 06_multimodal_cnn_lstm_training.ipynb con Papermill...
%JUPYTER_CMD% notebooks/06_multimodal_cnn_lstm_training.ipynb notebooks/06_multimodal_cnn_lstm_training.ipynb --cwd notebooks --log-output
if %errorlevel% neq 0 (
    echo.
    echo ERROR en 06_multimodal_cnn_lstm_training.ipynb.
) else (
    echo Finalizado 06_multimodal_cnn_lstm_training.ipynb con exito.
)
echo.

echo ============================================================
echo EJECUCION DE TODOS LOS MODELOS FINALIZADA
echo ============================================================
pause
