#ifndef OFFLINE_INTERPRETER_OPTIMALMEMPLANNER_H
#define OFFLINE_INTERPRETER_OPTIMALMEMPLANNER_H

#include "TensorPlanning.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_planner/memory_planner.h"

class OptimalMemPlanner : public tflite::MemoryPlanner {
 public:
  TfLiteStatus AddBuffer(tflite::ErrorReporter *error_reporter, int size,
                         int first_time_used, int last_time_used) override;

  size_t GetMaximumMemorySize() override;

  int GetBufferCount() override;

  TfLiteStatus GetOffsetForBuffer(tflite::ErrorReporter *error_reporter,
                                  int buffer_index, int *offset) override;

 private:
  void CalcIfNeeded();

 private:
  bool m_needCalc = true;
  struct BufferInfo {
    int index;
    int size;
    int first_use;
    int last_use;
  };
  std::vector<BufferInfo> m_bufferInfo;
};

#endif
