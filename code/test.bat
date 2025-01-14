
@echo off
setlocal enabledelayedexpansion

for /L %%i in (1,1,10) do (
    echo Cora 0 %%i
    py .\GAT.py -d Cora -e 0 -i %%i > tmp.txt
    echo Cora 1 %%i
    py .\GAT.py -d Cora -e 1 -i %%i >  tmp.txt
    echo Cite 0 %%i
    py .\GAT.py -d CiteSeer -e 0 -i %%i >  tmp.txt
    echo Cite 1 %%i
    py .\GAT.py -d CiteSeer -e 1 -i %%i >  tmp.txt
)
del /f /q "tmp.txt" 
echo Test Finished!
