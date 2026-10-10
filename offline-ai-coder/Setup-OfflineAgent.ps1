# Setup-OfflineAgent.ps1 - installs the offline coding agent stack.
# Idempotent: re-running only fetches or installs what is missing.
param([string]$InstallDir = "$env:LOCALAPPDATA\OfflineCodingAgent")
$ErrorActionPreference = 'Continue'
$Log = Join-Path $InstallDir 'setup.log'
function Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line
}
function Get-File($url, $out) {
    if (Test-Path $out) { Log "Already downloaded: $out"; return $true }
    Log "Downloading $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
        return $true
    } catch {
        Log "Download failed: $($_.Exception.Message)"
        return $false
    }
}
$Temp = Join-Path $env:TEMP 'OfflineAgentSetup'
New-Item -ItemType Directory -Force -Path $Temp | Out-Null
$fail = $false

# ---------------------------------------------------------------
# 1) Ollama runtime
# ---------------------------------------------------------------
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
    $exe = Join-Path $Temp 'OllamaSetup.exe'
    if (Get-File 'https://ollama.com/download/OllamaSetup.exe' $exe) {
        Log 'Installing Ollama (silent)...'
        Start-Process $exe -ArgumentList '/S' -Wait
    } else { $fail = $true }
}
if (Test-Path $ollama) {
    Log 'Starting the local model server...'
    $up = $false
    try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null; $up = $true } catch {}
    if (-not $up) {
        Start-Process $ollama -ArgumentList 'serve' -WindowStyle Hidden
        foreach ($i in 1..40) {
            Start-Sleep 3
            try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
        }
    }
    if ($up) { Log 'Model server is up on this PC (127.0.0.1:11434).' }
    else { Log 'Model server did not come up.'; $fail = $true }
} else {
    Log 'Ollama was not found after install.'
    $fail = $true
}

# ---------------------------------------------------------------
# 2) Models: 14B main coder, 7B fast, plus a 16K-context 14B
# ---------------------------------------------------------------
if (-not $fail) {
    $list = & $ollama list 2>$null | Out-String
    if ($list -notmatch 'qwen2\.5-coder:14b') {
        Log 'Pulling qwen2.5-coder:14b (about 9 GB, this is the slow part)...'
        & $ollama pull qwen2.5-coder:14b
    } else { Log 'qwen2.5-coder:14b already present.' }
    if ($list -notmatch 'qwen2\.5-coder:7b') {
        Log 'Pulling qwen2.5-coder:7b (about 4.7 GB)...'
        & $ollama pull qwen2.5-coder:7b
    } else { Log 'qwen2.5-coder:7b already present.' }
    $list = & $ollama list 2>$null | Out-String
    if ($list -notmatch 'qwen-coder-14b-16k') {
        $mf = Join-Path $Temp 'Modelfile'
        "FROM qwen2.5-coder:14b`nPARAMETER num_ctx 16384" | Set-Content -Path $mf -Encoding ASCII
        Log 'Creating qwen-coder-14b-16k (14B with a 16K context window)...'
        & $ollama create qwen-coder-14b-16k -f $mf
    } else { Log 'qwen-coder-14b-16k already present.' }
    Log 'Installed models:'
    & $ollama list | ForEach-Object { Log $_ }
}

# ---------------------------------------------------------------
# 3) VS Code + Cline extension
# ---------------------------------------------------------------
$codeCmd = "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd"
if (-not (Test-Path $codeCmd)) {
    $vs = Join-Path $Temp 'VSCodeUserSetup.exe'
    if (Get-File 'https://update.code.visualstudio.com/latest/win32-x64-user/stable' $vs) {
        Log 'Installing VS Code (silent)...'
        Start-Process $vs -ArgumentList '/VERYSILENT /NORESTART /MERGETASKS=!runcode' -Wait
    }
}
if (Test-Path $codeCmd) {
    Log 'Installing the Cline extension in VS Code...'
    & $codeCmd --install-extension saoudrizwan.claude-dev
} else {
    Log 'VS Code was not found, so Cline was skipped. Aider (below) still works.'
}

# ---------------------------------------------------------------
# 4) Python + Aider (terminal coding partner)
# ---------------------------------------------------------------
$python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $python)) {
    $py = Join-Path $Temp 'python-3.12.10-amd64.exe'
    if (Get-File 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' $py) {
        Log 'Installing Python 3.12 (silent)...'
        Start-Process $py -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1' -Wait
    }
}
if (Test-Path $python) {
    Log 'Installing Aider (this pulls a few hundred MB of Python packages)...'
    & $python -m pip install --upgrade pip aider-chat
    $aiderCfg = Join-Path $env:USERPROFILE '.aider.conf.yml'
    if (-not (Test-Path $aiderCfg)) {
        'model: ollama_chat/qwen-coder-14b-16k' | Set-Content -Path $aiderCfg -Encoding ASCII
        Log "Wrote Aider config: $aiderCfg"
    }
    [Environment]::SetEnvironmentVariable('OLLAMA_API_BASE', 'http://127.0.0.1:11434', 'User')
    $env:OLLAMA_API_BASE = 'http://127.0.0.1:11434'
} else {
    Log 'Python was not found, so Aider was skipped.'
}

# ---------------------------------------------------------------
# 5) Sample project for the airplane-mode test
# ---------------------------------------------------------------
$sampleSrc = Join-Path $InstallDir 'sample'
$sampleDst = Join-Path $env:USERPROFILE 'OfflineAgentTest'
if ((Test-Path $sampleSrc) -and -not (Test-Path $sampleDst)) {
    Copy-Item $sampleSrc $sampleDst -Recurse
    Log "Sample test project copied to $sampleDst"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Log 'Note: Git was not found. Aider still works; install Git later if you want change history.'
}

if ($fail) {
    Log 'FINISHED WITH ERRORS. Re-run this installer when you have internet; it picks up where it left off.'
    exit 1
}
Log 'All done. Open Start-Here.txt in the install folder for the 2-minute Cline finish and the offline test.'
exit 0
