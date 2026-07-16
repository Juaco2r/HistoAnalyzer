# Windows Inno Setup invocation fix — v1.0.8

The native Windows executable, OpenCV ML self-test, and portable ZIP completed successfully, but the installer step failed because `Get-Command iscc.exe` could return multiple command objects. Expanding `.FullName` on that collection produced an array, which PowerShell's call operator could not execute.

Version 1.0.8:

- selects exactly one `ISCC.exe` command result;
- resolves it to an absolute executable path;
- falls back to the standard Inno Setup installation paths;
- resolves `build/windows_installer.iss` to an absolute path;
- logs both resolved paths before execution;
- captures and validates the Inno Setup exit code.
