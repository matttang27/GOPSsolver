[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [switch]$SkipInstall,
    [switch]$SkipBuild,
    [switch]$KeepOutputs,
    [double]$ExploitabilityThreshold = 1e-8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ReportsDir = Join-Path $RepoRoot "reports"
$EphemeralOutputDir = $null

if ($KeepOutputs) {
    $OutputDir = $ReportsDir
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir | Out-Null
    }
}
else {
    if (-not (Test-Path $ReportsDir)) {
        New-Item -ItemType Directory -Path $ReportsDir | Out-Null
    }
    $EphemeralOutputDir = Join-Path $ReportsDir (".tmp_cpp_cache_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $EphemeralOutputDir | Out-Null
    $OutputDir = $EphemeralOutputDir
}

$WinCache = Join-Path $OutputDir "full6_ci.evc"
$PointsCache = Join-Path $OutputDir "full6points_ci.evc"
$WinJson = Join-Path $OutputDir "full6_ci.exploitability.json"
$PointsJson = Join-Path $OutputDir "full6points_ci.exploitability.json"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Assert-PythonDependenciesPresent {
    $probe = @"
import numpy
import pandas
import scipy
import ortools
import pyarrow
"@
    $probe | & python -
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependencies are missing. Run once with -InstallDeps."
    }
}

function Get-CppSolverPath {
    $candidates = @(
        (Join-Path $RepoRoot "solver\build\vcpkg\Release\cpp_solver.exe"),
        (Join-Path $RepoRoot "solver\build\vcpkg\Release\cpp_solver"),
        (Join-Path $RepoRoot "solver\build\Release\cpp_solver.exe"),
        (Join-Path $RepoRoot "solver\build\cpp_solver.exe"),
        (Join-Path $RepoRoot "solver\build\cpp_solver")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $buildDir = Join-Path $RepoRoot "solver\build"
    if (Test-Path $buildDir) {
        $fallback = Get-ChildItem -Path $buildDir -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.BaseName -eq "cpp_solver" } |
            Select-Object -First 1
        if ($fallback) {
            return $fallback.FullName
        }
    }
    throw "Could not locate cpp_solver binary."
}

Push-Location $RepoRoot
try {
    if ($InstallDeps -and $SkipInstall) {
        throw "Use either -InstallDeps or -SkipInstall, not both."
    }

    if ($InstallDeps) {
        Invoke-Step -Name "Install Python dependencies" -Command {
            & python -m pip install --upgrade pip
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
            & python -m pip install -r requirements.txt
        }
    } else {
        Write-Host "==> Skipping dependency install (use -InstallDeps to install)"
        Invoke-Step -Name "Verify Python dependencies" -Command {
            Assert-PythonDependenciesPresent
        }
    }

    if (-not $SkipBuild) {
        $presetPath = Join-Path $RepoRoot "solver\CMakePresets.json"
        if (Test-Path $presetPath) {
            Invoke-Step -Name "Configure cpp_solver (preset vcpkg)" -Command {
                Push-Location (Join-Path $RepoRoot "solver")
                try {
                    & cmake --preset vcpkg
                }
                finally {
                    Pop-Location
                }
            }
            Invoke-Step -Name "Build cpp_solver (preset vcpkg)" -Command {
                Push-Location (Join-Path $RepoRoot "solver")
                try {
                    & cmake --build --preset vcpkg
                }
                finally {
                    Pop-Location
                }
            }
        } else {
            Invoke-Step -Name "Configure cpp_solver" -Command {
                & cmake -S solver -B solver/build -DCMAKE_BUILD_TYPE=Release
            }
            Invoke-Step -Name "Build cpp_solver" -Command {
                & cmake --build solver/build --config Release --parallel
            }
        }
    } else {
        Write-Host "==> Skipping C++ build"
    }

    $solverExe = Get-CppSolverPath

    Invoke-Step -Name "Generate n=6 win cache" -Command {
        & $solverExe --n 6 --cache-out $WinCache
    }
    Invoke-Step -Name "Generate n=6 points cache" -Command {
        & $solverExe --n 6 --objective points --cache-out $PointsCache
    }

    Invoke-Step -Name "Run exploitability (win cache)" -Command {
        & python ai/exploitability.py --policy evc-ne --cache $WinCache --n 6 --json-out $WinJson
    }
    Invoke-Step -Name "Run exploitability (points cache)" -Command {
        & python ai/exploitability.py --policy evc-ne --cache $PointsCache --n 6 --json-out $PointsJson
    }

    Invoke-Step -Name "Assert exploitability thresholds" -Command {
        $script = @"
import json
from pathlib import Path

checks = [
    (Path(r"$WinJson"), "win"),
    (Path(r"$PointsJson"), "points"),
]
threshold = float($ExploitabilityThreshold)

for path, expected_obj in checks:
    data = json.loads(path.read_text(encoding="utf-8"))
    actual_obj = data.get("strategy_objective")
    if actual_obj != expected_obj:
        raise SystemExit(f"{path}: expected strategy_objective={expected_obj}, got {actual_obj}")
    exploitability = float(data["avg_max_exploitability"])
    if abs(exploitability) > threshold:
        raise SystemExit(f"{path}: avg_max_exploitability={exploitability} exceeds {threshold}")

print("Exploitability checks passed.")
"@
        $script | & python -
    }

    $env:GOPS_CPP_SOLVER = $solverExe
    Invoke-Step -Name "Run parity unittest" -Command {
        & python -m unittest tests.test_solver_parity_cpp_python -v
    }

    Write-Host ""
    Write-Host "Local workflow completed successfully."
    if ($KeepOutputs) {
        Write-Host "Generated files:"
        Write-Host "  $WinCache"
        Write-Host "  $PointsCache"
        Write-Host "  $WinJson"
        Write-Host "  $PointsJson"
    } else {
        Write-Host "Generated files are temporary and will be removed."
    }
}
finally {
    if ($EphemeralOutputDir -and (Test-Path $EphemeralOutputDir)) {
        Remove-Item -Path $EphemeralOutputDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
