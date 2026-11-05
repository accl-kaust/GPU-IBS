cmake_minimum_required(VERSION 3.18)
project(FastIBS VERSION 1.0.0)

# Set C++ standard and build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

# Version reported by --version: the project version, plus the git revision
# of the checkout when there is one (-DGPUIBS_REVISION=... overrides it,
# e.g. for a build from a source archive).
if(NOT DEFINED GPUIBS_REVISION)
    execute_process(COMMAND git describe --always --dirty
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE GPUIBS_REVISION
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
add_compile_definitions(GPUIBS_VERSION="${PROJECT_VERSION}" GPUIBS_REVISION="${GPUIBS_REVISION}")

# Ignore BOOST deprecated headers
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")
add_definitions("-DBOOST_TIMER_ENABLE_DEPRECATED")


# CMake 4 removed the bundled FindBoost module; use Boost's own config
# package (what conda / modern Boost installs provide).
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# Prefer pthread for Threads package
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost REQUIRED) # header-only use (boost/progress.hpp)


# Optional CUDA GPU probe path (Phase 1): -DFASTIBS_CUDA=ON
# The evaluation targets A100 (sm_80) and H100 (sm_90); V100 (sm_70) is
# included by default so the same binary also runs on mixed clusters where
# a job may land on an older node. Override with
# -DCMAKE_CUDA_ARCHITECTURES=... The binaries fall back to the CPU path at
# runtime when no device is present.
option(FASTIBS_CUDA "Build the CUDA GPU probe path" OFF)
if(FASTIBS_CUDA)
    # Must be set before enable_language(CUDA): that call defines
    # CMAKE_CUDA_ARCHITECTURES to nvcc's own default (sm_52), after which a
    # NOT DEFINED guard never fires and the requested architectures are
    # silently ignored. Only a -D on the command line pre-defines it here.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
    endif()
    # The shared host/device headers use C++17 (std::string_view); be
    # explicit rather than relying on the nvcc default dialect.
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    add_definitions("-DFASTIBS_CUDA")
    # The C++ translation units include <cuda_runtime.h> through the device
    # headers, so g++ needs the toolkit include directory too. It is not a
    # default search path when the toolkit lives in /usr/local/cuda (the
    # nvidia/cuda container images); CUDA::toolkit carries it.
    find_package(CUDAToolkit REQUIRED)
    set(FASTIBS_GPU_SOURCES gpu/DeviceKmerIndex.cu gpu/DeviceHashIndex.cu gpu/EncodeProbe.cu)
    set(FASTIBS_GPU_LIBS CUDA::toolkit)
else()
    set(FASTIBS_GPU_SOURCES "")
    set(FASTIBS_GPU_LIBS "")
endif()

# Vendored KMC random-access API, built from source (the tools need only
# the API, not libkmc_core.a, so no Docker/KMC build step is required).
add_library(kmc_api STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../KMC/kmc_api/kmc_file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../KMC/kmc_api/kmer_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../KMC/kmc_api/mmer.cpp)

# Add executable targets
add_executable(fastibs FastIBS.cpp ${FASTIBS_GPU_SOURCES})
add_executable(fastibsmapper FastIBSMapper.cpp ${FASTIBS_GPU_SOURCES})
add_executable(KDBIntersect KDBIntersect.cpp ${FASTIBS_GPU_SOURCES})

# Link libraries to the executable targets
target_link_libraries(fastibs PRIVATE ZLIB::ZLIB Threads::Threads Boost::boost kmc_api ${FASTIBS_GPU_LIBS})
target_link_libraries(fastibsmapper PRIVATE ZLIB::ZLIB Threads::Threads Boost::boost kmc_api ${FASTIBS_GPU_LIBS})
target_link_libraries(KDBIntersect PRIVATE ZLIB::ZLIB Threads::Threads Boost::boost kmc_api ${FASTIBS_GPU_LIBS})

if(FASTIBS_CUDA)
    add_executable(probe_bench gpu/ProbeBench.cu gpu/DeviceKmerIndex.cu gpu/DeviceHashIndex.cu)
    target_link_libraries(probe_bench PRIVATE Threads::Threads)
endif()

# cmake --install build [--prefix ~/.local]  ->  <prefix>/bin/fastibs ...
include(GNUInstallDirs)
install(TARGETS fastibs fastibsmapper KDBIntersect RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(FASTIBS_CUDA)
    install(TARGETS probe_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()



# Package information
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
