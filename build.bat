@echo off
REM SimSwap Docker Build Script for Windows
REM Usage: build.bat [tag] [--no-cache]

setlocal enabledelayedexpansion

REM Default values
set TAG=simswap-api:latest
set NO_CACHE=

REM Parse arguments
if "%~1" neq "" set TAG=%~1
if "%~2" equ "--no-cache" set NO_CACHE=--no-cache

echo 🚀 Building SimSwap Docker Image...
echo Tag: %TAG%
echo No Cache: %NO_CACHE%
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    exit /b 1
)

REM Check if NVIDIA Docker is available
docker run --rm --gpus all nvidia/cuda:11.1-base-ubuntu20.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Warning: NVIDIA Docker not available. GPU support may not work.
    echo    Continue building anyway...
    echo.
)

REM Start build
echo 📦 Starting build process...
set start_time=%time%

REM Build command
if "%NO_CACHE%" equ "--no-cache" (
    docker build --no-cache -t "%TAG%" .
) else (
    docker build -t "%TAG%" .
)

REM Check build result
if errorlevel 1 (
    echo.
    echo ❌ Build failed!
    echo Please check the error messages above.
    exit /b 1
)

echo.
echo ✅ Build successful!
echo ⏱️  Build completed at %time%
echo.

REM Show image info
echo 📊 Image Information:
docker images | findstr simswap-api
echo.

REM Test image
echo 🧪 Testing image...
docker run --rm --gpus all -d --name simswap-test -p 8000:8000 "%TAG%" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Could not start container for testing
) else (
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:8000/health >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  Image started but health check failed
    ) else (
        echo ✅ Image test passed!
    )
    docker stop simswap-test >nul 2>&1
)

echo.
echo 🎉 Build completed successfully!
echo.
echo To run the container:
echo   docker run --gpus all -p 8000:8000 %TAG%
echo.
echo Or use Docker Compose:
echo   docker-compose up -d

pause
