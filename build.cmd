REM Updating git repo

:: @echo off
:: set /p message="Enter git commit message: "

:: git add -A
:: git commit -m %message%
:: git push

REM Cleaning up before building.
git clean -fdx

REM Building the python package.
python setup.py sdist bdist_wheel

REM Uploading with twine.
twine upload dist/*

timeout 15
