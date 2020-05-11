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

## Usage from target code

    extern void Setup();
    extern void Eval();
    extern void *GetInputPtr();
    extern const void *GetOutputPtr();

    int main()
    {
        Setup();
        // For single float in/out such as the TFLM sine example.
        *(float*)GetInputPtr() = in;
        Eval();
        float out = *(float*)GetOutputPtr();
    }

## TODO

This project is a work on progress. Important open points:

- Properly link to TF Lite
- Implement OptimalMemPlanner
- Fix possible ABI incompatibilities
