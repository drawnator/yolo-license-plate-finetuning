# Full three-stage training on Windows with Docker Desktop (no own WSL distro needed).
# Windows counterpart of run_full.sh, sized by default for a 16 GB GPU.
#
#   Start / continue after a crash:  .\run_full.ps1
#   Stop:                            docker stop yolo-train
#
# Nothing else has to be typed: the log is written to runs\logs\yolo-train.log and
# metrics plus CPU/RAM/GPU usage go to MLflow. Re-running is safe: --resume continues an
# interrupted stage from its last.pt (saved every epoch under angelicam\full\).
#
# Requirements: Docker Desktop with the "WSL 2 based engine" enabled (needed for the GPU),
# a recent NVIDIA driver, and the datasets\ folder copied next to this script.
#
# Settings. Override any of them before the call, e.g.:
#   $env:BATCH = 6; $env:FINAL_BATCH = 16; .\run_full.ps1
# The 16 GB numbers are scaled from measurements on an 8 GB card (see
# hardware_requirements\bh02_training.md) and were not run on a 16 GB card. In the first
# minutes check the GPU_mem column in the log: above ~14G, lower the batch.

function Setting($name, $default) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($value)) { $default } else { $value }
}

$Name       = Setting 'TRAIN_NAME'    'yolo-train'
$Project    = Setting 'TRAIN_PROJECT' 'angelicam/full'
$Batch      = Setting 'BATCH'         '8'     # yolo26l stages (pretrain + pseudo-label teachers)
$FinalBatch = Setting 'FINAL_BATCH'   '16'    # yolo26s final stage; 24 should fit (estimate, not measured)
$TeacherEpochs = Setting 'TEACHER_EPOCHS' '30'  # pseudo-label teachers start from the pretrain
$Workers    = Setting 'WORKERS'       '8'     # raise to 12-16 on a CPU with that many cores
$Memory     = Setting 'MEMORY'        '44g'   # must fit inside Docker Desktop's memory limit
$MemorySwap = Setting 'MEMORY_SWAP'   '60g'
$ExtraArgs  = Setting 'EXTRA_ARGS'    ''
$MlflowUri  = Setting 'MLFLOW_TRACKING_URI' 'https://mlflow.guilherme.zip'
$MlflowUser = Setting 'MLFLOW_TRACKING_USERNAME' 'admin'
$MlflowPass = Setting 'MLFLOW_TRACKING_PASSWORD' ''

Set-Location $PSScriptRoot
$Here = (Get-Location).Path

foreach ($dir in 'datasets', 'datasets\label_sets', 'runs\logs', 'angelicam') {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$running = docker inspect -f '{{.State.Running}}' $Name 2>$null
if ($running -eq 'true') {
    Write-Host "$Name is already running. Stop it first: docker stop $Name"
    exit 1
}

docker build -t yolo-train:latest -f docker/Dockerfile .
if ($LASTEXITCODE -ne 0) { Write-Host 'Image build failed.'; exit 1 }
docker rm $Name 2>$null | Out-Null

$commit = git rev-parse HEAD 2>$null
if (-not $commit) { $commit = 'unknown' }

# No double quotes inside this string: Windows PowerShell mangles them for native commands.
$train = "set -o pipefail; python -m training.train_pipeline " +
    "--synthetic 50000 --pretrain --skip-export --mix-synthetic " +
    "--project $Project " +
    "--batch-size $Batch --final-batch-size $FinalBatch --teacher-epochs $TeacherEpochs " +
    "--imgsz 640 --workers $Workers --cache off " +
    "--resume $ExtraArgs " +
    "2>&1 | tee -a /runs/logs/$Name.log"

docker run -d --gpus all --memory=$Memory --memory-swap=$MemorySwap --shm-size=8g `
    --name $Name `
    -e PYTHONUNBUFFERED=1 `
    -e PYTHONDONTWRITEBYTECODE=1 `
    -e MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true `
    -e "MLFLOW_TRACKING_URI=$MlflowUri" `
    -e "MLFLOW_TRACKING_USERNAME=$MlflowUser" `
    -e "MLFLOW_TRACKING_PASSWORD=$MlflowPass" `
    -e "GIT_COMMIT_HASH=$commit" `
    -v "${Here}\datasets:/datasets" `
    -v "${Here}\runs:/runs" `
    -v "${Here}\angelicam:/angelicam" `
    -v "${Here}\datasets\label_sets:/datasets/label_sets" `
    yolo-train:latest bash -c $train
if ($LASTEXITCODE -ne 0) { Write-Host 'Container failed to start.'; exit 1 }

Write-Host "Started $Name. Log: runs\logs\$Name.log"
