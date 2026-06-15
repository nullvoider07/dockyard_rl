#!/usr/bin/env bash
# Generate the control-center gRPC stubs from the vendored proto.
# Run at image build time (build-time grpcio-tools is allowed; runtime needs
# only grpcio). protoc emits an absolute `import control_center_pb2` in the
# _grpc module; rewrite it to a package-relative import so the stubs import as
# dockyard_rl.experience.cua.clients.proto.*.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m grpc_tools.protoc \
  -I "$HERE" \
  --python_out="$HERE" \
  --grpc_python_out="$HERE" \
  "$HERE/control_center.proto"

# Make the _grpc module's sibling import package-relative.
sed -i 's/^import control_center_pb2 as/from . import control_center_pb2 as/' \
  "$HERE/control_center_pb2_grpc.py"

echo "Generated control_center_pb2.py + control_center_pb2_grpc.py in $HERE"
