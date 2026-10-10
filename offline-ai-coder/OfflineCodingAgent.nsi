; OfflineCodingAgent-Setup.nsi - per-user installer for the offline coding agent.
; Compiled with makensis (NSIS 3.x). No admin rights required:
; everything installs under %LOCALAPPDATA%.

!include "MUI2.nsh"
!include "LogicLib.nsh"

Name "Offline Coding Agent"
OutFile "/home/hatch/workspace/your_files/OfflineCodingAgent-Setup.exe"
InstallDir "$LOCALAPPDATA\OfflineCodingAgent"
RequestExecutionLevel user
SetCompressor /SOLID lzma
ShowInstDetails show
BrandingText "Offline Coding Agent Setup"

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TEXT "This wizard sets up your offline coding agent - no administrator rights needed.$\r$\n$\r$\nIt downloads about 15 GB on first run (the Ollama model runner, the Qwen2.5-Coder 14B and 7B models, VS Code with Cline, and Aider), so the install step takes a while on slower connections. That is the one and only time it needs the internet: after setup, the whole stack runs 100% offline. Re-running the installer later only fetches what is missing.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN "$INSTDIR\Open-StartHere.bat"
!define MUI_FINISHPAGE_RUN_TEXT "Open the Start-Here notes"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Offline Coding Agent" SecMain
  SetOutPath "$INSTDIR"
  File /r "stage\*"

  DetailPrint "Installing the offline stack (about 15 GB of downloads on first run)..."
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\Setup-OfflineAgent.ps1" -InstallDir "$INSTDIR"'
  Pop $0
  ${If} $0 != "0"
    MessageBox MB_ICONEXCLAMATION|MB_OK "Some parts did not finish installing (exit code $0).$\n$\nYour files were copied, and you can re-run this installer once you have an internet connection - it picks up where it left off. Details are in $INSTDIR\setup.log."
  ${EndIf}

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\Offline Coding Agent"
  CreateShortcut "$SMPROGRAMS\Offline Coding Agent\Aider (Offline Coder).lnk" "$INSTDIR\Aider-Offline.bat" "" "$INSTDIR\Aider-Offline.bat" 0 SW_SHOWNORMAL "" "$INSTDIR"
  CreateShortcut "$SMPROGRAMS\Offline Coding Agent\Run Offline Test.lnk" "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" '-NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\Run-OfflineTest.ps1"' "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" 0 SW_SHOWNORMAL "" "$INSTDIR"
  CreateShortcut "$SMPROGRAMS\Offline Coding Agent\Start-Here Notes.lnk" "$WINDIR\notepad.exe" '"$INSTDIR\Start-Here.txt"' "$WINDIR\notepad.exe" 0 SW_SHOWNORMAL "" "$INSTDIR"
  CreateShortcut "$SMPROGRAMS\Offline Coding Agent\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\Aider (Offline Coder).lnk" "$INSTDIR\Aider-Offline.bat" "" "$INSTDIR\Aider-Offline.bat" 0 SW_SHOWNORMAL "" "$INSTDIR"
  CreateShortcut "$DESKTOP\Run Offline Test.lnk" "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" '-NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\Run-OfflineTest.ps1"' "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" 0 SW_SHOWNORMAL "" "$INSTDIR"

  ; Add/Remove Programs entry (per-user)
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent" "DisplayName" "Offline Coding Agent"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent" "Publisher" "Infusion"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent" "DisplayVersion" "1.0"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\Aider (Offline Coder).lnk"
  Delete "$DESKTOP\Run Offline Test.lnk"
  Delete "$SMPROGRAMS\Offline Coding Agent\Aider (Offline Coder).lnk"
  Delete "$SMPROGRAMS\Offline Coding Agent\Run Offline Test.lnk"
  Delete "$SMPROGRAMS\Offline Coding Agent\Start-Here Notes.lnk"
  Delete "$SMPROGRAMS\Offline Coding Agent\Uninstall.lnk"
  RMDir "$SMPROGRAMS\Offline Coding Agent"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\OfflineCodingAgent"
SectionEnd
