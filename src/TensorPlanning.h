#ifndef OFFLINE_INTERPRETER_TENSORPLANNING_H
#define OFFLINE_INTERPRETER_TENSORPLANNING_H

#include <vector>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/tensor_utils.h"

namespace tflite {
class MicroInterpreter;
}

struct TensorLifetime {
  int firstUse;
  int lastUse;
  bool needsAlloc;
};

std::vector<TensorLifetime> GetTensorLifetimes(
    tflite::MicroInterpreter *interpreter);

#endif
