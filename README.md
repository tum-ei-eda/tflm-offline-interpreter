# TensorFlow Lite for Micro - Offline Interpreter

This project runs the TFLM interpreter off-target to produce static code that is able to run inference on the processed model. This avoids interpretation and code size overheads on a target that does not need to process different models dynamically.

## Dependencies

- CMake 3.13
- Tensorflow 2.2

An always matching TF fork is provided. TF Lite dependencies need to be downloaded first:

    git clone https://github.com/tum-ei-eda/tensorflow.git
    cd tensorflow
    git checkout tumeda
    cd tensorflow/lite/tools/make
    ./download_dependencies.sh

## Building

    mkdir build && cd build
    cmake -DTF_SRC=/path/to/tf ..
    make

## Running

    ./tflm-offline-interpreter modelFile.tflite outFile.cpp

## TODO

This project is a work on progress. Important open points:

- Properly link to TF Lite
- Implement OptimalMemPlanner
- Fix possible ABI incompatibilities
