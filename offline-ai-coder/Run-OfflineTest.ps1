# Run-OfflineTest.ps1 - proves the coding agent works with the internet off.
$ollama  = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$aider   = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\aider.exe"
$python  = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$testDir = Join-Path $env:USERPROFILE 'OfflineAgentTest'

Write-Host ''
Write-Host '================ OFFLINE (AIRPLANE MODE) TEST ================'
Write-Host 'Step 1: Turn ON airplane mode now - Wi-Fi off, cable unplugged.'
Read-Host 'Press Enter once airplane mode is on'

$online = $false
try { Invoke-WebRequest 'https://example.com' -TimeoutSec 5 -UseBasicParsing | Out-Null; $online = $true } catch {}
if ($online) {
    Write-Host 'Heads up: the internet still answers, so airplane mode may not be on. The test still runs.'
} else {
    Write-Host 'Confirmed: no internet. Everything below runs only on this PC.'
}

Write-Host ''
Write-Host 'Step 2: Your local models (these live on this PC):'
& $ollama list

Write-Host ''
Write-Host 'Step 3: Aider opens the sample project. When it starts, paste this task:'
Write-Host '    Implement reverse_words in stringtools.py so all tests in test_stringtools.py pass. Do not change the tests.'
Write-Host 'Aider shows you every change it makes. When it finishes, type  /exit'
Read-Host 'Press Enter to open Aider'
Set-Location $testDir
$env:OLLAMA_API_BASE = 'http://127.0.0.1:11434'
if (Test-Path $aider) { & $aider stringtools.py test_stringtools.py } else { aider stringtools.py test_stringtools.py }

Write-Host ''
Write-Host 'Step 4: Running the tests against the edited code...'
& $python -m unittest -v

Write-Host ''
Write-Host 'If the tests pass, your offline coding agent just planned, edited,'
Write-Host 'and verified code with the internet off. That is the whole test.'
Read-Host 'Press Enter to close'
