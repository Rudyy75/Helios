#!/bin/bash
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$SCRIPT_DIR/.."

echo "Starting Helios server..."
exec "$ROOT_DIR/build/Release/helios_server" "$ROOT_DIR/models/resnet50.onnx"
