@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "WINSDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
set "PATH=%VCToolsInstallDir%bin\Hostx64\x64;%WINSDK_BIN%;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set "WINSDK_BIN_FWD=%WINSDK_BIN:\=/%"

cd /d D:\D_backup\2025\tum\26w\Ithome\cASE
if exist case_2\adaptive_publish_tmp\build_m7_clean rmdir /s /q case_2\adaptive_publish_tmp\build_m7_clean

cmake -S case_2\adaptive_publish_tmp -B case_2\adaptive_publish_tmp\build_m7_clean -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_RC_COMPILER="%WINSDK_BIN_FWD%/rc.exe" -DCMAKE_MT="%WINSDK_BIN_FWD%/mt.exe" -DADAPTIVEGEMM_BUILD_PYTHON=OFF
if errorlevel 1 exit /b 1

cmake --build case_2\adaptive_publish_tmp\build_m7_clean --config Release
exit /b %ERRORLEVEL%
