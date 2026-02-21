REM Ghost-Shell Bot Launcher
REM This VBS script launches the bot with a visible command prompt window

Set objFSO = CreateObject("Scripting.FileSystemObject")
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

REM Run the batch file with visible window
Set objShell = CreateObject("WScript.Shell")
strBatchFile = strScriptPath & "\run_bot.bat"

REM Launch command prompt with the batch file
REM Parameter 1 = visible window (normal size)
REM Parameter False = don't wait for process to finish (so VBS closes immediately)
objShell.Run "cmd.exe /k """ & strBatchFile & """", 1, False
