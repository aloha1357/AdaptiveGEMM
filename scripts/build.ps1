[CmdletBinding()]
param(
    [string]$BuildDirectory = 'build',
    [ValidateSet('Release', 'Debug')][string]$Configuration = 'Release',
    [switch]$Python,
    [switch]$SkipTests,
    [string]$TestRegex = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $projectRoot $BuildDirectory
}

if (-not $env:VCToolsInstallDir -or -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio C++ build tools, or run from an x64 developer shell.' }
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installation) { throw 'Visual Studio x64 C++ tools were not found.' }
    $vcvars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
    $cmdLine = "call `"$vcvars`" >nul && set"
    $environmentLines = & $env:ComSpec /d /s /c $cmdLine
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment initialization failed.' }
    foreach ($line in $environmentLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

# Some VS installations omit the SDK executables from PATH even though the
# headers/libraries are configured. Discover the installed SDK, without pinning
# a version or a user's workspace path.
if (-not (Get-Command rc.exe -ErrorAction SilentlyContinue) -or -not (Get-Command mt.exe -ErrorAction SilentlyContinue)) {
    $sdkRoot = if ($env:WindowsSdkDir) { $env:WindowsSdkDir } else { Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10' }
    $sdkBin = Get-ChildItem -LiteralPath (Join-Path $sdkRoot 'bin') -Directory |
        Where-Object { (Test-Path -LiteralPath (Join-Path $_.FullName 'x64/rc.exe')) -and (Test-Path -LiteralPath (Join-Path $_.FullName 'x64/mt.exe')) } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if (-not $sdkBin) { throw 'Install the Windows SDK resource compiler and manifest tool.' }
    $env:PATH = "$(Join-Path $sdkBin.FullName 'x64');$env:PATH"
}

if ($env:CUDA_PATH) {
    $env:PATH = "$(Join-Path $env:CUDA_PATH 'bin/x64');$(Join-Path $env:CUDA_PATH 'bin');$env:PATH"
}
$binding = if ($Python) { 'ON' } else { 'OFF' }
$resourceCompiler = (Get-Command rc.exe).Source -replace '\\', '/'
$manifestTool = (Get-Command mt.exe).Source -replace '\\', '/'
# Native tools report warnings on stderr; use their exit status for failures.
$ErrorActionPreference = 'Continue'
& cmake -S $projectRoot -B $BuildDirectory -G 'NMake Makefiles' "-DCMAKE_BUILD_TYPE=$Configuration" "-DADAPTIVEGEMM_BUILD_PYTHON=$binding" "-DCMAKE_RC_COMPILER=$resourceCompiler" "-DCMAKE_MT=$manifestTool"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& cmake --build $BuildDirectory --config $Configuration
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
if (-not $SkipTests) {
    $testArguments = @('--test-dir', $BuildDirectory, '-C', $Configuration, '--output-on-failure', '--timeout', '180')
    if ($TestRegex) { $testArguments += @('-R', $TestRegex) }
    & ctest @testArguments
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed.' }
}
