
@echo off
setlocal enabledelayedexpansion

for /L %%i in (1,1,10) do (
    py .\GAT.py -d Cora -e 0 -i %%i
    py .\GAT.py -d Cora -e 1 -i %%i
    py .\GAT.py -d CiteSeer -e 0 -i %%i
    py .\GAT.py -d CiteSeer -e 1 -i %%i
)

echo Test Finished!
