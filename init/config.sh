# config.sh -- settings read by install.sh, scripts/build_img.sh, scripts/start_cont.sh

IMAGE_NAME="fastibs_img"        # Docker image name (runtime stage; fastibs.sif is built from it)
CONTAINER_NAME="fastibs_cont"   # Docker container name (the compile container)

# ---- GPU-IBS additions. All optional; the defaults give a GPU-capable build.
BUILDER_IMAGE_NAME="${IMAGE_NAME}_build"   # toolchain image the compile container runs
SIF_NAME="${SIF_NAME:-fastibs.sif}"        # Singularity/Apptainer image written by install.sh (run.sh and
                                           # intersect.sh also take the self-contained gpu-ibs.sif: SIF_NAME=gpu-ibs.sif)

# FASTIBS_CUDA: auto (default) builds the GPU path when nvcc is present in the
# build environment; on forces it; off builds CPU-only binaries and a CPU-only
# toolchain image.
FASTIBS_CUDA="${FASTIBS_CUDA:-auto}"
CUDA_ARCHS="${CUDA_ARCHS:-70;80;90}"       # V100, A100, H100

BUILDER_BASE="${BUILDER_BASE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
RUNTIME_BASE="${RUNTIME_BASE:-ubuntu:22.04}"
if [ "${FASTIBS_CUDA}" = "off" ]; then
    BUILDER_BASE="ubuntu:22.04"
fi

# Container runtime for .sif images: apptainer when installed, else singularity.
CONTAINER_CMD="${CONTAINER_CMD:-$(command -v apptainer >/dev/null 2>&1 && echo apptainer || echo singularity)}"
