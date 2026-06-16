@echo off
setlocal enabledelayedexpansion

:: Force Rust/Cargo to always output colors and live progress.
set CARGO_TERM_COLOR=always

:: Anchor to the script's actual directory.
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

:: -----------------------------------------------------
:: Strata Native / Slint Workspace Builder
::
:: Double-click (or run with no arguments) for an interactive numbered menu.
:: Or pass arguments directly to skip the prompt (scriptable):
::
:: Usage:
::   build-windows.bat [mode] [flags...]
::
:: Modes:
::   dev | debug  - cargo build               (default; fast local build)
::   release      - cargo build --release     (optimized, no console window)
::   run          - cargo run                 (dev build + launch)
::   check        - cargo check
::   clean        - cargo clean
::
:: Flags (any order, after the mode):
::   onnx         - enable the Parallax Studio's ML depth (DepthAnything via
::                  ONNX Runtime). Adds the `depth-onnx` feature, builds the
::                  strata-desktop package, and copies the real DirectML.dll
::                  next to the .exe so it runs on double-click.
::
:: Examples:
::   build-windows.bat                 (dev, heuristic depth only)
::   build-windows.bat release         (release, no ML depth)
::   build-windows.bat release onnx    (release WITH DepthAnything)
::   build-windows.bat dev onnx        (dev WITH DepthAnything)
::   build-windows.bat run onnx
:: -----------------------------------------------------

set "ONNX=0"

:: No arguments (e.g. double-clicked in Explorer) -> show an interactive menu.
:: Arguments provided -> use them directly (scriptable, no prompt).
if "%~1"=="" goto menu

set "MODE=%~1"
if /I "%MODE%"=="debug" set "MODE=dev"
if /I "%MODE%"=="development" set "MODE=dev"
for %%A in (%*) do (
    if /I "%%~A"=="onnx" set "ONNX=1"
    if /I "%%~A"=="--onnx" set "ONNX=1"
    if /I "%%~A"=="depth-onnx" set "ONNX=1"
)
goto have_mode

:menu
echo ===================================================
echo   Strata Engine - Build Menu
echo ===================================================
echo   1^) Dev build               (fast, no ML depth)
echo   2^) Dev build + ONNX        (DepthAnything)
echo   3^) Release build           (optimized, no ML depth)
echo   4^) Release build + ONNX    (optimized, DepthAnything)
echo   5^) Run                     (dev build then launch)
echo   6^) Run + ONNX
echo   7^) Check                   (type-check only)
echo   8^) Clean                   (remove build artifacts)
echo.
set /p "CHOICE=Enter choice [1-8] (default 1): "
if "%CHOICE%"=="" set "CHOICE=1"
if "%CHOICE%"=="1" ( set "MODE=dev"     & set "ONNX=0" )
if "%CHOICE%"=="2" ( set "MODE=dev"     & set "ONNX=1" )
if "%CHOICE%"=="3" ( set "MODE=release" & set "ONNX=0" )
if "%CHOICE%"=="4" ( set "MODE=release" & set "ONNX=1" )
if "%CHOICE%"=="5" ( set "MODE=run"     & set "ONNX=0" )
if "%CHOICE%"=="6" ( set "MODE=run"     & set "ONNX=1" )
if "%CHOICE%"=="7" ( set "MODE=check"   & set "ONNX=0" )
if "%CHOICE%"=="8" ( set "MODE=clean"   & set "ONNX=0" )
if not defined MODE (
    echo [ERROR] Invalid choice "%CHOICE%".
    goto end_error
)
echo.

:have_mode
if not exist "Cargo.toml" (
    echo [ERROR] No root Cargo.toml discovered at %PROJECT_ROOT%.
    goto end_error
)

:: ONNX builds must target the strata-desktop package + feature.
set "FEATURE_ARGS="
if "%ONNX%"=="1" set "FEATURE_ARGS=-p strata-desktop --features depth-onnx"

set "CARGO_CMD="
set "OUTPUT_DIR="
set "SUCCESS_LABEL="

if /I "%MODE%"=="dev" (
    set "CARGO_CMD=cargo build %FEATURE_ARGS%"
    set "OUTPUT_DIR=target\debug\"
    set "SUCCESS_LABEL=Development build complete"
    goto run_cargo
)
if /I "%MODE%"=="release" (
    set "CARGO_CMD=cargo build --release %FEATURE_ARGS%"
    set "OUTPUT_DIR=target\release\"
    set "SUCCESS_LABEL=Release build complete"
    goto run_cargo
)
if /I "%MODE%"=="run" (
    set "CARGO_CMD=cargo run %FEATURE_ARGS%"
    set "OUTPUT_DIR=target\debug\"
    set "SUCCESS_LABEL=Run finished"
    goto run_cargo
)
if /I "%MODE%"=="check" (
    set "CARGO_CMD=cargo check %FEATURE_ARGS%"
    set "OUTPUT_DIR=target\debug\"
    set "SUCCESS_LABEL=Workspace check complete"
    goto run_cargo
)
if /I "%MODE%"=="clean" (
    set "CARGO_CMD=cargo clean"
    set "OUTPUT_DIR=target\"
    set "SUCCESS_LABEL=Clean complete"
    goto run_cargo
)

echo [ERROR] Unknown mode "%MODE%".
echo.
echo Usage: build-windows.bat [dev^|release^|run^|check^|clean] [onnx]
goto end_error

:run_cargo
echo ===================================================
echo Strata Engine - Native / Slint Workspace Builder
echo ===================================================
echo Project Root : %PROJECT_ROOT%
echo Mode         : %MODE%
if "%ONNX%"=="1" (
    echo Depth        : DepthAnything ONNX ^(depth-onnx^)
) else (
    echo Depth        : heuristic only ^(no ML^)
)
echo Command      : %CARGO_CMD%
echo.

%CARGO_CMD%
if %ERRORLEVEL% neq 0 goto end_error

:: ONNX builds: the pyke ORT prebuilt for Windows depends on DirectML.dll at
:: load time, and cargo places it as a symlink into its cache (which Windows
:: can't resolve on double-click). Replace it with a real copy next to the exe.
if "%ONNX%"=="1" if not "%MODE%"=="clean" if not "%MODE%"=="check" (
    echo Copying real DirectML.dll next to the executable...
    powershell -NoProfile -Command ^
      "$dst = Join-Path '%OUTPUT_DIR%' 'DirectML.dll';" ^
      "$it = Get-Item $dst -ErrorAction SilentlyContinue;" ^
      "if ($it -and $it.LinkType) { $t = $it.Target; if ($t -is [array]) { $t = $t[0] }; Remove-Item $dst -Force; Copy-Item $t $dst -Force; Write-Host '  -> copied real DirectML.dll' }" ^
      "elseif ($it) { Write-Host '  -> DirectML.dll already a real file' }" ^
      "else { Write-Host '  -> DirectML.dll not found (build may be cached; touch a source file and rebuild)' }"
)

echo.
echo ===================================================
echo [SUCCESS] %SUCCESS_LABEL%!
echo Output folder: %OUTPUT_DIR%
echo ===================================================
goto end_success

:end_error
echo.
echo ===================================================
echo [FAILED] The compilation script ran into an error.
echo ===================================================
pause
exit /b 1

:end_success
echo.
pause
