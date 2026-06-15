# Overlay on the user's ubuntu-desktop-code CUA image that installs the OSWorld
# guest server on :5000 (Phase 2, evaluator/setup bridge "option b"). With it,
# the control-center backend's vendored evaluators and task `config[]` setup
# steps run unchanged against the same HTTP contract the official backend uses,
# while actuation stays on control-center (:50051) and observation on The-Eye
# (:8080). Only the guest server's getter/setup/file endpoints are used here.
#
# Build (needs a host with the base image pulled; cannot be validated offline):
#   docker build -f osworld-guest.dockerfile \
#     --build-arg BASE_IMAGE=nullvoider/cua-ubuntu-24.04-code:v0.1 \
#     -t cua-ubuntu-24.04-code-osworld:v0.1 .

ARG BASE_IMAGE=nullvoider/cua-ubuntu-24.04-code:v0.1
FROM ${BASE_IMAGE}

# Pinned to the same OSWorld commit as vendor/desktop_env_eval so the guest-side
# getters/setup and the host-side evaluators match exactly.
ARG OSWORLD_COMMIT=705623ca18e0055dd995fd5a350d6588cff2caf5

USER root

# Guest server runtime deps. xdotool is already present (control-center uses it);
# the GI/AT-SPI stack powers the optional /accessibility endpoint, scrot/tk back
# pyautogui+Pillow screenshots on X11.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git scrot python3-tk python3-dev python3-xlib \
      gir1.2-atspi-2.0 python3-gi python3-pyatspi \
    && rm -rf /var/lib/apt/lists/*

# Fetch only the pinned guest server (sparse, blobless).
RUN git clone --filter=blob:none --sparse https://github.com/xlang-ai/OSWorld /opt/osworld \
    && cd /opt/osworld \
    && git sparse-checkout set desktop_env/server \
    && git checkout ${OSWORLD_COMMIT}

# Guest server Python deps (match its requirements; pynput/pygame/pywinauto are
# not needed for the getter/setup/file endpoints this backend uses).
RUN pip install --no-cache-dir \
      "PyAutoGUI==0.9.54" Pillow flask numpy lxml requests python-xlib

# Run the guest server inside the session user's X/desktop context on :5000.
COPY osworld-guest.service /etc/systemd/system/osworld-guest.service
RUN ln -sf /etc/systemd/system/osworld-guest.service \
      /etc/systemd/system/multi-user.target.wants/osworld-guest.service

EXPOSE 5000
