#!/bin/bash
# Builds the Docker images. Run from init/ (install.sh does): config.sh and
# the Dockerfile live there, and init/ is the build context -- the project is
# bind-mounted into the container, never copied into an image.
set -e

# Read config values
source config.sh

echo "##############################"
echo "Building toolchain image ${BUILDER_IMAGE_NAME} (from ${BUILDER_BASE}) ..."
docker build --target builder -t "${BUILDER_IMAGE_NAME}" \
    --build-arg BUILDER_BASE="${BUILDER_BASE}" \
    --build-arg RUNTIME_BASE="${RUNTIME_BASE}" . #--no-cache

echo "Building runtime image ${IMAGE_NAME} (from ${RUNTIME_BASE}) ..."
docker build --target runtime -t "${IMAGE_NAME}" \
    --build-arg BUILDER_BASE="${BUILDER_BASE}" \
    --build-arg RUNTIME_BASE="${RUNTIME_BASE}" .
echo "##############################"
