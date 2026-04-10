#!/bin/bash

set -euxo pipefail

rm -rf build || true

# get torch libraries for osx-arm64
LIBTORCH_DIR=${BUILD_PREFIX}
if [[ "$OSTYPE" == "darwin"* && $OSX_ARCH == "arm64" ]]; then
    LIBTORCH_DIR=${RECIPE_DIR}/libtorch
    conda list -p ${BUILD_PREFIX} > packages.txt
    cat packages.txt
    PYTORCH_PACKAGE_VERSION=`grep pytorch packages.txt | awk -F ' ' '{print $2}'`
    CONDA_SUBDIR=osx-arm64 conda create -y -p ${LIBTORCH_DIR} --no-deps pytorch=${PYTORCH_PACKAGE_VERSION} python=${PY_VER}
fi

echo "testing"
echo "prefix: $PREFIX"
echo "${LIBTORCH_DIR}"
echo `ls -d $PREFIX`


test="123"
echo "$test"

CMAKE_FLAGS="  -DCMAKE_INSTALL_PREFIX=${PREFIX}"
CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Release"

CMAKE_FLAGS+=" -DBUILD_TESTING=ON"
CMAKE_FLAGS+=" -DOPENMM_DIR=${PREFIX}"
CMAKE_FLAGS+=" -DPYTORCH_DIR=${SP_DIR}/torch"
CMAKE_FLAGS+=" -DTorch_DIR=${LIBTORCH_DIR}/lib/python${PY_VER}/site-packages/torch/share/cmake/Torch"
# OpenCL
CMAKE_FLAGS+=" -DNN_BUILD_OPENCL_LIB=ON"
CMAKE_FLAGS+=" -DOPENCL_INCLUDE_DIR=${PREFIX}/include"
CMAKE_FLAGS+=" -DOPENCL_LIBRARY=${PREFIX}/lib/libOpenCL${SHLIB_EXT}"

CMAKE_FLAGS+=" -DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES="
CMAKE_FLAGS+=" -DCMAKE_CUDA_COMPILER=${PREFIX}/bin/nvcc"

CMAKE_FLAGS+=" -DPYTHON_EXECUTABLE=${PYTHON}"
declare -a CUDA_CONFIG_ARGS

if [ ${cuda_compiler_version} != "None" ]; then
    ARCH_LIST=$(${PYTHON} -c "import torch; print(';'.join([f'{y[0]}.{y[1]}' for y in [x[3:] for x in torch._C._cuda_getArchFlags().split() if x.startswith('sm_')]]))")
    CMAKE_FLAGS+=" -DTORCH_CUDA_ARCH_LIST=${ARCH_LIST}"
    if [ majorversion ${cuda_compiler_version} -ge 12 ]; then
	# This is required because conda-forge stores cuda headers in a non standard location
	export CUDA_INC_PATH=$CONDA_PREFIX/$targetsDir/include
    fi
else
    CMAKE_FLAGS+=" -DENABLE_CUDA=OFF"
fi

# Build in subdirectory and install.
mkdir -p build
cd build
cmake ${CMAKE_ARGS} ${CMAKE_FLAGS} ${SRC_DIR}
#cmake ${CMAKE_FLAGS} ${SRC_DIR}
make -j$CPU_COUNT install
make -j$CPU_COUNT PythonInstall

if [[ "$OSTYPE" == "darwin"* && $OSX_ARCH == "arm64" ]]; then
    # clean up, otherwise, environment is stored in package
    rm -fr ${LIBTORCH_DIR}
fi

echo `find $PREFIX | grep openmmnapshift`