param(
    [string]$buildType
)

# Check if the build type is valid
if ($buildType -ne "RELEASE" -and $buildType -ne "DEBUG") {
    Write-Host "Invalid build type. Please use 'RELEASE' or 'DEBUG'."
    exit 1
}

# Set LIBTORCH environment variable based on build type
if ($buildType -eq "RELEASE") {
    $libtorchPath = "C:\Users\kchin\Code\Kaiyotech\LIBTORCH\Release\libtorch"
} else {
    $libtorchPath = "C:\Users\kchin\Code\Kaiyotech\LIBTORCH\Debug\libtorch"
}

# Set or overwrite LIBTORCH environment variable
[System.Environment]::SetEnvironmentVariable("LIBTORCH", $libtorchPath)

# Append lib subdirectory to LIBTORCH environment variable
$libPath = Join-Path $libtorchPath "lib"
$existingPath = [Environment]::GetEnvironmentVariable("PATH")
$newPath = "$existingPath;$libPath"
[System.Environment]::SetEnvironmentVariable("PATH", $newPath)

Write-Host "Environment variables updated for $buildType."
