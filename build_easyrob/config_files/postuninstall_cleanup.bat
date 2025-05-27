@echo off
setlocal

:: Get the installation directory (the folder where this .bat is located)
set "INSTALL_DIR=%~dp0"

:: Create a secondary cleanup script in the temporary folder
set "CLEANUP=%TEMP%\_cleanup_%RANDOM%.bat"

(
    echo @echo off
    echo timeout /t 1 ^>nul
    echo rd /s /q "%INSTALL_DIR%" 2^>nul
    echo del "%%~f0"
) > "%CLEANUP%"

:: Start the secondary cleanup script
start /min cmd /c "%CLEANUP%"

endlocal
exit /b 0
