#!/bin/bash
# Top-level installer -- the same steps as the original FastIBS installer:
#   1. build the Docker images (toolchain + runtime)
#   2. start the compile container with this directory mounted at /project
#   3. compile the binaries inside it (they land in ./bin)
#   4. save the runtime image as a .tar
#   5. build fastibs.sif from the .tar with apptainer or singularity
#   6. remove the .tar
# Without Docker but with apptainer/singularity, fastibs.sif is built from
# init/fastibs.def instead and the binaries are compiled inside it.
set -e

# Navigate to the root directory of the project
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# Read config values
source init/config.sh

have_docker=0
if command -v docker >/dev/null 2>&1; then
    # A docker client without a reachable daemon (or without permission to
    # use it) is common on HPC login nodes; fall through to apptainer then.
    if docker info >/dev/null 2>&1; then
        have_docker=1
    else
        echo "docker is installed but not usable here (daemon unreachable or no permission)" >&2
    fi
fi

if [ "${have_docker}" = 1 ]; then
    # Build Docker images and start container
    (cd init && bash ../scripts/build_img.sh && bash ../scripts/start_cont.sh)

    # Compile the project, as the invoking user: build/ and bin/ then belong
    # to you on the host (compiled as root they could not be removed or
    # rebuilt by a later native compile.sh)
    docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp \
        -e FASTIBS_CUDA="${FASTIBS_CUDA}" -e CUDA_ARCHS="${CUDA_ARCHS}" \
        "${CONTAINER_NAME}" bash -c "cd /project && bash compile.sh"

    # Build docker image tar
    echo "  ** saving docker image ${IMAGE_NAME} into a tar file **  "
    docker save -o "${IMAGE_NAME}.tar" "${IMAGE_NAME}:latest"

    # Build Singularity/Apptainer image
    if command -v "${CONTAINER_CMD}" >/dev/null 2>&1; then
        "${CONTAINER_CMD}" build --force "${SIF_NAME}" "docker-archive://${IMAGE_NAME}.tar"
    else
        echo "warning: neither apptainer nor singularity found; not building ${SIF_NAME}" >&2
        echo "         (the Docker image ${IMAGE_NAME} and the binaries in ./bin are ready)" >&2
    fi

    # remove tar
    rm -f "${IMAGE_NAME}.tar"
elif command -v "${CONTAINER_CMD}" >/dev/null 2>&1; then
    echo "building ${SIF_NAME} with ${CONTAINER_CMD} from init/fastibs.def (no usable docker)"
    fakeroot=()
    if [ "$(id -u)" -ne 0 ]; then
        fakeroot=(--fakeroot)
    fi
    # init/fastibs.def names the default toolchain base; honour BUILDER_BASE
    # from init/config.sh (ubuntu:22.04 for FASTIBS_CUDA=off) by building from
    # a copy with the From: line replaced.
    tmpdef=$(mktemp --suffix=.def)
    sed "s|^From: .*|From: ${BUILDER_BASE}|" init/fastibs.def > "${tmpdef}"
    "${CONTAINER_CMD}" build "${fakeroot[@]}" --force "${SIF_NAME}" "${tmpdef}"
    rm -f "${tmpdef}"
    # --cleanenv: compile from the image's own environment, as `docker exec`
    # does; otherwise an active conda env on the host (CC/CXX/CXXFLAGS,
    # CONDA_PREFIX) would steer CMake to host compilers and libraries.
    "${CONTAINER_CMD}" exec --cleanenv --bind "$(pwd)":/project "${SIF_NAME}" \
        bash -c "cd /project && FASTIBS_CUDA='${FASTIBS_CUDA}' CUDA_ARCHS='${CUDA_ARCHS}' bash compile.sh"
else
    echo "error: neither docker nor apptainer/singularity is installed" >&2
    exit 1
fi

if [ -f "${SIF_NAME}" ]; then
    echo "Installation complete: binaries in ./bin, container image ${SIF_NAME}"
else
    echo "Installation complete: binaries in ./bin (no ${SIF_NAME} built)"
fi
