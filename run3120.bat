@echo off
REM Save current directory
set "orig_dir=%CD%"

REM Change to Python 3.7.6 environment directory
cd /d C:\Envs\py3120env

REM Activate the virtual environment (adjust based on your env setup)
call Scripts\activate.bat

REM Return to original directory
cd /d "%orig_dir%"

REM Pause so you can see the result
cmd
