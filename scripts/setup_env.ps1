param(
    [string]$VcpkgDir = "$PSScriptRoot\..\vcpkg"
)

Write-Host "Setting up vcpkg at $VcpkgDir..."

if (-Not (Test-Path -Path $VcpkgDir)) {
    git clone https://github.com/microsoft/vcpkg.git $VcpkgDir
    Set-Location -Path $VcpkgDir
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install
} else {
    Write-Host "vcpkg already exists at $VcpkgDir"
    Set-Location -Path $VcpkgDir
}

Write-Host "Installing dependencies..."
.\vcpkg install onnxruntime:x64-windows
.\vcpkg install crow:x64-windows
.\vcpkg install nlohmann-json:x64-windows
.\vcpkg install gtest:x64-windows

Write-Host "Done!"
