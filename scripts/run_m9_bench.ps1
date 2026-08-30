# M9 SchemeII_Fused benchmark matrix.
# Runs one GPU benchmark at a time. Do not launch this script twice in parallel.
#
# Usage (from workspace root D:\D_backup\2025\tum\26w\Ithome\cASE):
#   powershell -ExecutionPolicy Bypass -File case_2\adaptive_publish_tmp\scripts\run_m9_bench.ps1
#   powershell -ExecutionPolicy Bypass -File case_2\adaptive_publish_tmp\scripts\run_m9_bench.ps1 -Variant 3 -Label after_tile
#   powershell -ExecutionPolicy Bypass -File case_2\adaptive_publish_tmp\scripts\run_m9_bench.ps1 -CorrectnessOnly
#
# Documented rectangular correctness:
#   .\build_m9_opt_ninja\adaptive_gemm_scheme_ii_fused_compare_test.exe -M 768 -N 640 -K 1000 --precision_policy Auto --num_moduli 0 --scheme_ii_backend custom_mma --scheme_ii_group_size 4 --use_true_fused_kernel

param(
    [string]$BuildDir = "case_2\adaptive_publish_tmp\build_m9_opt_ninja",
    [string]$OutDir = "case_2\adaptive_publish_tmp",
    [string]$Label = "m9",
    [int]$Variant = 0,
    [string]$PrologueFusion = "none",
    [switch]$SkipLarge,
    [switch]$CorrectnessOnly,
    [switch]$IncludeControls
)

$ErrorActionPreference = "Stop"
$inner = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $inner
if (Test-Path (Join-Path (Get-Location) $BuildDir)) {
    $root = (Get-Location).Path
} elseif (Test-Path (Join-Path $workspace $BuildDir)) {
    $root = $workspace
} elseif (Test-Path (Join-Path $inner ($BuildDir.Split('\')[-1]))) {
    $root = $inner
    $BuildDir = Split-Path -Leaf $BuildDir
} else {
    $root = $workspace
}
Set-Location $root

$bench = Join-Path $BuildDir "adaptive_gemm_precision_test.exe"
$cmp = Join-Path $BuildDir "adaptive_gemm_scheme_ii_fused_compare_test.exe"
if (-not (Test-Path $bench)) { throw "Missing $bench" }
if (-not (Test-Path $cmp)) { throw "Missing $cmp" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvPath = Join-Path $OutDir ("m9_bench_{0}_{1}.csv" -f $Label, $stamp)
$txtPath = Join-Path $OutDir ("m9_bench_{0}_{1}.txt" -f $Label, $stamp)

function Get-Field([string]$text, [string]$name) {
    if ($text -match "$name=([^\s]+)") { return $Matches[1] }
    return ""
}

function Invoke-One {
    param([string]$Exe, [string[]]$Args, [string]$Title)
    Write-Host ""
    Write-Host "===== $Title ====="
    Write-Host ("CMD: {0} {1}" -f $Exe, ($Args -join " "))
    $out = & $Exe @Args 2>&1 | Out-String
    Write-Host $out
    return $out
}

$header = "label,M,N,K,num_moduli,scheme_ii_group_size,used_partial_crt_epilogue,used_true_fused_kernel,scheme_ii_gemm_launches,scheme_ii_mma_variant,scheme_ii_prologue_fusion,schemeII_fused_custom_mma_ms,schemeII_fused_cublas_backend_ms,cublas13_dynamic_emu_ms,cublas13_fixed53_emu_ms,native_dgemm_ms,max_abs_custom_vs_schemeII,max_abs_custom_vs_cublas_backend,fallback_mode"
$lines = @($header)
$txt = @()
$txt += "M9 benchmark $Label $(Get-Date -Format o)"
$txt += "BuildDir=$BuildDir Variant=$Variant PrologueFusion=$PrologueFusion"
$txt += "GPU benchmarks were run sequentially, not in parallel."
$txt += ""

$fusionArgs = @()
if ($PrologueFusion -and $PrologueFusion -ne "none") {
    $fusionArgs = @("--scheme_ii_prologue_fusion", $PrologueFusion)
}
$variantArgs = @()
if ($Variant -ne 0) {
    $variantArgs = @("--scheme_ii_mma_variant", "$Variant")
}

function Add-CompareRow([int]$M, [int]$N, [int]$K, [string]$Title, [string[]]$Extra) {
    $args = @("-M", "$M", "-N", "$N", "-K", "$K", "--precision_policy", "Auto", "--num_moduli", "0", "--scheme_ii_backend", "custom_mma") + $Extra
    $out = Invoke-One $cmp $args $Title
    $row = [pscustomobject]@{
        label = $Label
        M = $M; N = $N; K = $K
        num_moduli = Get-Field $out "num_moduli"
        scheme_ii_group_size = Get-Field $out "scheme_ii_group_size"
        used_partial_crt_epilogue = Get-Field $out "used_partial_crt_epilogue"
        used_true_fused_kernel = Get-Field $out "used_true_fused_kernel"
        scheme_ii_gemm_launches = Get-Field $out "scheme_ii_gemm_launches"
        scheme_ii_mma_variant = Get-Field $out "scheme_ii_mma_variant"
        scheme_ii_prologue_fusion = Get-Field $out "scheme_ii_prologue_fusion"
        schemeII_fused_custom_mma_ms = ""
        schemeII_fused_cublas_backend_ms = ""
        cublas13_dynamic_emu_ms = ""
        cublas13_fixed53_emu_ms = ""
        native_dgemm_ms = ""
        max_abs_custom_vs_schemeII = Get-Field $out "max_abs_custom_vs_schemeII"
        max_abs_custom_vs_cublas_backend = Get-Field $out "max_abs_custom_vs_cublas_backend"
        fallback_mode = Get-Field $out "fallback_mode"
        passed = ($out -match "TEST PASSED")
    }
    $script:txt += $Title
    $script:txt += $out.Trim()
    $script:txt += ""
    return $row
}

function Add-BenchRow([int]$M, [int]$N, [int]$K, [int]$Repeat, [string]$Title, [string[]]$Extra) {
    $args = @("-M", "$M", "-N", "$N", "-K", "$K", "--precision_policy", "Auto", "--num_moduli", "0", "--repeat", "$Repeat", "--benchmark", "--scheme_ii_backend", "custom_mma") + $Extra
    $out = Invoke-One $bench $args $Title
    $row = [pscustomobject]@{
        label = $Label
        M = $M; N = $N; K = $K
        num_moduli = Get-Field $out "num_moduli"
        scheme_ii_group_size = Get-Field $out "scheme_ii_group_size"
        used_partial_crt_epilogue = Get-Field $out "used_partial_crt_epilogue"
        used_true_fused_kernel = Get-Field $out "used_true_fused_kernel"
        scheme_ii_gemm_launches = Get-Field $out "scheme_ii_gemm_launches"
        scheme_ii_mma_variant = Get-Field $out "scheme_ii_mma_variant"
        scheme_ii_prologue_fusion = Get-Field $out "scheme_ii_prologue_fusion"
        schemeII_fused_custom_mma_ms = Get-Field $out "schemeII_fused_custom_mma_ms"
        schemeII_fused_cublas_backend_ms = Get-Field $out "schemeII_fused_cublas_backend_ms"
        cublas13_dynamic_emu_ms = Get-Field $out "cublas13_dynamic_emu_ms"
        cublas13_fixed53_emu_ms = Get-Field $out "cublas13_fixed53_emu_ms"
        native_dgemm_ms = Get-Field $out "cublas_fp64_ms"
        max_abs_custom_vs_schemeII = Get-Field $out "max_abs_custom_vs_schemeII"
        max_abs_custom_vs_cublas_backend = Get-Field $out "max_abs_custom_vs_cublas_backend"
        fallback_mode = Get-Field $out "fallback_mode"
        passed = $true
    }
    $script:txt += $Title
    $script:txt += $out.Trim()
    $script:txt += ""
    return $row
}

function Write-Row($row) {
    $script:lines += ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}" -f `
        $row.label, $row.M, $row.N, $row.K, $row.num_moduli, $row.scheme_ii_group_size, `
        $row.used_partial_crt_epilogue, $row.used_true_fused_kernel, $row.scheme_ii_gemm_launches, `
        $row.scheme_ii_mma_variant, $row.scheme_ii_prologue_fusion, `
        $row.schemeII_fused_custom_mma_ms, $row.schemeII_fused_cublas_backend_ms, `
        $row.cublas13_dynamic_emu_ms, $row.cublas13_fixed53_emu_ms, $row.native_dgemm_ms, `
        $row.max_abs_custom_vs_schemeII, $row.max_abs_custom_vs_cublas_backend, $row.fallback_mode)
}

$speedExtra = @("--scheme_ii_group_size", "4", "--use_true_fused_kernel") + $variantArgs + $fusionArgs

Write-Row (Add-CompareRow 512 512 512 "correctness 512^3 M8.3" $speedExtra)
Write-Row (Add-CompareRow 768 640 1000 "correctness 768x640x1000 M8.3" $speedExtra)

if (-not $CorrectnessOnly) {
    Write-Row (Add-BenchRow 1024 1024 1024 7 "bench 1024^3 M8.3" $speedExtra)
    if (-not $SkipLarge) {
        Write-Row (Add-BenchRow 2048 2048 2048 3 "bench 2048^3 M8.3" $speedExtra)
        Write-Row (Add-BenchRow 4096 4096 4096 2 "bench 4096^3 M8.3" $speedExtra)
        Write-Row (Add-CompareRow 4096 4096 4096 "correctness 4096^3 M8.3" $speedExtra)
    }
    if ($IncludeControls) {
        Write-Row (Add-BenchRow 1024 1024 1024 7 "control 1024^3 M8.4 true fused" @("--scheme_ii_group_size", "16", "--use_true_fused_kernel"))
        Write-Row (Add-BenchRow 1024 1024 1024 7 "control 1024^3 M8.2 grouped no partial" @("--scheme_ii_group_size", "4"))
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$lines | Set-Content -Encoding utf8 $csvPath
$txt | Set-Content -Encoding utf8 $txtPath
Write-Host ""
Write-Host "CSV: $csvPath"
Write-Host "TXT: $txtPath"
