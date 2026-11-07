# ================================================================
#  Du lieu tho -> .jsonl -> tokenizer moi -> train.bin / eval.bin
#
#  Chay tu thu muc goc project (sau setup_gpu_vm.ps1):
#    powershell -ExecutionPolicy Bypass -File scripts\windows\build_pretrain_data.ps1
#
#  Chay thu nho (khong dung toi file that):
#    ... -MaxDocs 1000 -WorkDir <thu muc tam>
# ================================================================
param(
    [int]$VocabSize = 32000,
    [double]$TokenizerMbPerFile = 300,
    [int]$MaxDocs = 0,
    [string]$RawDir = "dataset\raw",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Loi (ma $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

$Python = ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Chua co $Python - chay setup_gpu_vm.ps1 truoc." }

# WorkDir rong = ghi vao vi tri that cua project.
if ($WorkDir) {
    $JsonlDir = Join-Path $WorkDir "jsonl"
    $Tokenizer = Join-Path (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force $WorkDir)) "tokenizer.json"
    $TokensDir = Join-Path $WorkDir "tokens"
} else {
    $JsonlDir = "dataset\pretrain\jsonl"
    $Tokenizer = "tokenizer\tokenizer.json"
    $TokensDir = "dataset\pretrain\tokens"
}

Write-Host "=== 1. Gom du lieu -> $JsonlDir ==="
$corpusArgs = @("-m", "scripts.build_pretrain_corpus", "--raw-dir", $RawDir, "--out-dir", $JsonlDir)
if ($MaxDocs -gt 0) { $corpusArgs += @("--max-docs", "$MaxDocs") }
Invoke-Checked $Python $corpusArgs

Write-Host "=== 2. Train tokenizer (vocab $VocabSize, toi da $TokenizerMbPerFile MB moi nguon) ==="
$trainFiles = Get-ChildItem (Join-Path $JsonlDir "train") -Filter *.jsonl | Sort-Object Name | ForEach-Object { $_.FullName }
Invoke-Checked $Python (@("-m", "tokenizer.train_tokenizer", "--vocab-size", "$VocabSize",
    "--max-mb-per-file", "$TokenizerMbPerFile", "--out", $Tokenizer, "--files") + $trainFiles)

Write-Host "=== 3. Tokenize train / eval -> $TokensDir ==="
foreach ($split in @("train", "eval")) {
    Invoke-Checked $Python @("-m", "scripts.prepare_pretrain", "--tokenizer", $Tokenizer,
        "--in-dir", (Join-Path $JsonlDir $split), "--out-prefix", (Join-Path $TokensDir $split))
}

Write-Host "=== 4. Kiem tra tokenizer + goi y max_steps ==="
Invoke-Checked $Python @("-m", "scripts.check_pretrain_data", "--tokenizer", $Tokenizer,
    "--tokens-dir", $TokensDir, "--train-config", "configs\pretrain_50m.json")

Write-Host "[OK] Xong. Nho sua vocab_size trong configs\model_50m.json = $VocabSize va max_steps theo goi y tren."
