# gRPC stubs for the control-center ControlService.
#
# control_center.proto is vendored (pinned to ../control-center/Proto). The
# generated control_center_pb2.py / control_center_pb2_grpc.py are NOT checked
# in — they are produced at image-build time by generate.sh (build-time
# grpcio-tools is allowed; runtime needs only grpcio). clients/control_center.py
# guard-imports them, so this package loads offline without the generated files.
