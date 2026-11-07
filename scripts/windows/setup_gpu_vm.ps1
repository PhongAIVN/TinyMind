# ================================================================
#  Cai dat may GPU thue (Windows): venv + torch CUDA + tai du lieu tho.
#
#  Chay tu thu muc goc project:
#    powershell -ExecutionPolicy Bypass -File scripts\windows\setup_gpu_vm.ps1
#    ... -SkipDownload      chi cai moi truong
#    ... -DryRun            chi liet ke file se tai, khong tai
#
#  Du lieu tai ve: dataset\raw\<nguon>\  (khop SOURCES trong scripts\build_pretrain_corpus.py)
# ================================================================
param(
    # cu128: can cho GPU Blackwell (RTX 50xx), van chay tren 30xx/40xx
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipDownload,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Loi (ma $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

# Moi dong: repo tren Hugging Face, thu muc trong dataset\raw, mau file can tai.
$Downloads = @(
    @{ Repo = "epfml/FineWeb2-HQ";                                     Dir = "fineweb_vi";      Include = @("vie_Latn/000_0000[0-8].parquet") },
    @{ Repo = "wikimedia/wikipedia";                                   Dir = "wiki_vi";         Include = @("20231101.vi/*.parquet") },
    @{ Repo = "vilm/OpenOrca-Viet";                                    Dir = "openorca_vi";     Include = @("*.jsonl") },
    @{ Repo = "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated";           Dir = "alpaca_gpt4_vi";  Include = @("*.json") },
    @{ Repo = "bkai-foundation-models/vi-alpaca";                      Dir = "vi_alpaca";       Include = @("data/*.parquet") },
    @{ Repo = "5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated"; Dir = "metamath_vi"; Include = @("*.json") },
    @{ Repo = "taidng/UIT-ViQuAD2.0";                                  Dir = "viquad";          Include = @("data/train-*.parquet") },
    @{ Repo = "PeterPaker123/vietnamese-code-dataset";                 Dir = "code_vi";         Include = @("data/*.parquet") },
    @{ Repo = "iamtarun/python_code_instructions_18k_alpaca";          Dir = "python_instruct"; Include = @("data/*.parquet") },
    @{ Repo = "OpenCoder-LLM/opc-annealing-corpus";                    Dir = "opc_code";        Include = @("synthetic_qa/1*.jsonl") }
)

$VenvPython = ".venv\Scripts\python.exe"

if (-not $DryRun) {
    Write-Host "=== 1. Python + venv ==="
    if (-not (Test-Path $VenvPython)) {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) { throw "Chua co python tren PATH. Cai Python 3.11 (tick 'Add to PATH') roi chay lai." }
        Invoke-Checked "python" @("-c", "import sys; assert sys.version_info >= (3, 10), sys.version")
        Invoke-Checked "python" @("-m", "venv", ".venv")
    }

    Write-Host "=== 2. torch CUDA + thu vien ==="
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-U", "pip")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "torch", "--index-url", $TorchIndex)
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "requirements.txt")
    Invoke-Checked $VenvPython @("-c", "import torch; assert torch.cuda.is_available(), 'torch khong thay GPU'; print(torch.__version__, torch.cuda.get_device_name(0))")
}

if ($SkipDownload) { Write-Host "[OK] Bo qua tai du lieu."; exit 0 }

Write-Host "=== 3. Tai du lieu tho ==="
$Hf = if (Test-Path ".venv\Scripts\hf.exe") { ".venv\Scripts\hf.exe" } else { "hf" }
foreach ($item in $Downloads) {
    $arguments = @("download", $item.Repo, "--repo-type", "dataset", "--local-dir", "dataset\raw\$($item.Dir)")
    foreach ($pattern in $item.Include) { $arguments += @("--include", $pattern) }
    if ($DryRun) { $arguments += "--dry-run" }
    Write-Host "--- $($item.Repo) -> dataset\raw\$($item.Dir)"
    Invoke-Checked $Hf $arguments
}

Write-Host "[OK] Xong. Buoc tiep: scripts\windows\build_pretrain_data.ps1"
