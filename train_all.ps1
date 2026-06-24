$ErrorActionPreference = "Stop"

$env:PYTHONUTF8=1

Write-Host "Training 1.5B model..."
.\env_slm\Scripts\python.exe -X utf8 backend\scripts\train_from_feedback.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --output_dir "models/finetuned_slm_1_5B" --batch_size 2
if ($LASTEXITCODE -ne 0) { throw "1.5B training failed" }

Write-Host "Training 3B model..."
.\env_slm\Scripts\python.exe -X utf8 backend\scripts\train_from_feedback.py --model_name "Qwen/Qwen2.5-3B-Instruct" --output_dir "models/finetuned_slm_3B" --batch_size 1
if ($LASTEXITCODE -ne 0) { throw "3B training failed" }

Write-Host "Both trainings completed successfully."
