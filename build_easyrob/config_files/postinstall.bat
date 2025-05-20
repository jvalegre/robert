@echo off
cd /d "%~dp0"

echo Hello from postinstall > C:\postinstall_test.txt


echo [+] Starting postinstall... > postinstall.log 2>&1
mkdir _internal\robert_env >> postinstall.log 2>&1
tar -xzf robert_env.tar.gz -C _internal\robert_env >> postinstall.log 2>&1
call _internal\robert_env\Scripts\activate.bat >> postinstall.log 2>&1
call _internal\robert_env\Scripts\conda-unpack >> postinstall.log 2>&1
echo [+] Done. >> postinstall.log 2>&1

