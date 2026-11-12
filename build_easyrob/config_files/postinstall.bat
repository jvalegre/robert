@echo off
cd /d "%~dp0"

call _internal\robert_env\Scripts\conda-unpack.exe >> postinstall.log 2>&1
echo [+] Done. >> postinstall.log 2>&1

