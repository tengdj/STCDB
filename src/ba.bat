@echo off
del /f /s /q "%USERPROFILE%\AppData\Local\GitHubDesktop\*.*"
rd /s /q "%USERPROFILE%\AppData\Local\GitHubDesktop"

del /f /s /q "%USERPROFILE%\AppData\Roaming\GitHub Desktop\*.*""
rd /s /q "%USERPROFILE%\AppData\Roaming\GitHub Desktop"
pause
