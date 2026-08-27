#!/bin/bash
set -e

VCPKG_DIR="$(dirname "$0")/../vcpkg"

echo "Setting up vcpkg at $VcpkgDir..."

if [ ! -d "$VCPKG_DIR" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"
    cd "$VCPKG_DIR"
    ./bootstrap-vcpkg.sh
else
    echo "vcpkg already exists at $VCPKG_DIR"
    cd "$VCPKG_DIR"
fi

echo "Installing dependencies..."
./vcpkg install onnxruntime
./vcpkg install crow
./vcpkg install nlohmann-json
./vcpkg install gtest

echo "Done! You can now run:"
echo "cmake -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake"
echo "cmake --build build --config Release"
