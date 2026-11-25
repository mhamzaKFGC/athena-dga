# Overnight chain: full-UMUDGA training -> unseen-family experiment -> Athena pipeline -> XAI -> final report.
# Each step logs to logs\overnight_*.log; a step failure is recorded and the chain continues.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$status = "logs\overnight_status.txt"
function Step($name, $cmd) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  START $name" | Add-Content $status
    $log = "logs\overnight_$name.log"
    & cmd /c "$cmd > `"$log`" 2>&1"
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  END   $name (exit $LASTEXITCODE)" | Add-Content $status
}
"" | Set-Content $status
Step "1_full_training" "python -u compare.py --full"
Step "2_holdout_training" "python -u compare.py --holdout suppobox_1,matsnu,gozi_nasa,nymaim,necurs,dyre,ramnit,locky,pykspa,banjori --tag _holdout"
Step "3_holdout_report" "python -u experiments\holdout_report.py"
Step "4_athena_pipeline" "python -u ..\pipeline\run_pipeline.py"
Step "5_xai" "python -u xai\run_xai.py --perm 16 --per-family 4 --legit 200"
Step "6_final_report" "python -u report_final.py"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  ALL DONE" | Add-Content $status
