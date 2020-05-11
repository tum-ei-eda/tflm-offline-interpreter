#include "TensorPlanning.h"

#include <sstream>
#define private public
#include "tensorflow/lite/micro/micro_interpreter.h"
#undef private

// This code is copied from micro_allocator.cc because its in an anon namespace.

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"

namespace tflite {

namespace {
// Used to hold information used during allocation calculations.
struct AllocationInfo {
  size_t bytes;
  int first_created;
  int last_used;
  bool needs_allocating;
  void** output_ptr;
};

// We align tensor buffers to 16-byte boundaries, since this is a common
// requirement for SIMD extensions.
constexpr int kBufferAlignment = 16;

class MicroBuiltinDataAllocator : public BuiltinDataAllocator {
 public:
  explicit MicroBuiltinDataAllocator(SimpleMemoryAllocator* memory_allocator)
      : memory_allocator_(memory_allocator) {}

  void* Allocate(size_t size, size_t alignment_hint) override {
    return memory_allocator_->AllocateFromTail(size, alignment_hint);
  }
  void Deallocate(void* data) override {
    // Do not deallocate, builtin data needs to be available for the life time
    // of the model.
  }

 private:
  SimpleMemoryAllocator* memory_allocator_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

TfLiteStatus AllocateVariables(
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* flatbuffer_tensors,
    TfLiteTensor* runtime_tensors, SimpleMemoryAllocator* allocator) {
  for (size_t i = 0; i < flatbuffer_tensors->size(); ++i) {
    if (flatbuffer_tensors->Get(i)->is_variable()) {
      runtime_tensors[i].data.data = allocator->AllocateFromTail(
          runtime_tensors[i].bytes, kBufferAlignment);
      // Allocation failure.
      if (runtime_tensors[i].data.data == nullptr) {
        return kTfLiteError;
      }
    }
    tflite::ResetVariableTensor(&(runtime_tensors[i]));
  }
  return kTfLiteOk;
}

// A helper class to construct AllocationInfo array. This array contains the
// lifetime of tensors / scratch_buffer and will be used to calculate the memory
// plan. Methods need to be called in order from `Init`, `Add*`, to `Finish`.
class AllocationInfoBuilder {
 public:
  AllocationInfoBuilder(ErrorReporter* reporter,
                        SimpleMemoryAllocator* allocator)
      : reporter_(reporter), allocator_(allocator) {}

  // Initializes the builder by allocating AllocationInfo array from the
  // simple memory allocator.
  TfLiteStatus Init(size_t tensor_count, size_t scratch_buffer_count) {
    tensor_count_ = tensor_count;
    buffer_count_ = scratch_buffer_count;
    return Allocate();
  }

  // Add allocaiton information for the tensors.
  TfLiteStatus AddTensors(const SubGraph* subgraph,
                          TfLiteTensor* runtime_tensors);
  // Add allocation information for the scratch buffers.
  TfLiteStatus AddScratchBuffers(internal::ScratchBufferHandle* buffer_handles);

  // Returns a pointer to the built AllocationInfo array.
  const AllocationInfo* Finish() const { return info_; }
  size_t Size() const { return tensor_count_ + buffer_count_; }

 private:
  // Allocate the output AllocationInfo array from the allocator_;
  TfLiteStatus Allocate();

  ErrorReporter* reporter_ = nullptr;
  SimpleMemoryAllocator* allocator_ = nullptr;
  size_t tensor_count_ = 0;
  size_t buffer_count_ = 0;
  AllocationInfo* info_ = nullptr;
};

TfLiteStatus AllocationInfoBuilder::Allocate() {
  size_t bytes = sizeof(AllocationInfo) * Size();
  info_ = reinterpret_cast<AllocationInfo*>(
      allocator_->AllocateFromTail(bytes, alignof(AllocationInfo)));
  if (info_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        reporter_,
        "Failed to allocate memory for allocation_info, %d bytes required",
        bytes);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus AllocationInfoBuilder::AddTensors(const SubGraph* subgraph,
                                               TfLiteTensor* runtime_tensors) {
  // Set up allocation info for all tensors.
  for (size_t i = 0; i < tensor_count_; ++i) {
    AllocationInfo* current = &info_[i];
    // TfLiteTensor.uint8 field is deprecated so use .data field instead.
    current->output_ptr = &(runtime_tensors[i].data.data);
    current->bytes = runtime_tensors[i].bytes;
    current->first_created = -1;
    current->last_used = -1;
    current->needs_allocating = (runtime_tensors[i].data.data == nullptr) &&
                                (!subgraph->tensors()->Get(i)->is_variable());
  }

  for (size_t i = 0; i < subgraph->inputs()->size(); ++i) {
    const int tensor_index = subgraph->inputs()->Get(i);
    AllocationInfo* current = &info_[tensor_index];
    current->first_created = 0;
  }

  // Mark all outputs as persistent to the end of the invocation.
  for (size_t i = 0; i < subgraph->outputs()->size(); ++i) {
    const int tensor_index = subgraph->outputs()->Get(i);
    AllocationInfo* current = &info_[tensor_index];
    current->last_used = subgraph->operators()->size() - 1;
  }

  // Figure out when the first and last use of each tensor is.
  for (int i = (subgraph->operators()->size() - 1); i >= 0; --i) {
    const auto* op = subgraph->operators()->Get(i);
    for (size_t n = 0; n < op->inputs()->size(); ++n) {
      const int tensor_index = op->inputs()->Get(n);
      AllocationInfo* current = &info_[tensor_index];
      if (((current->last_used == -1) || (current->last_used < i))) {
        current->last_used = i;
      }
    }
    for (size_t n = 0; n < op->outputs()->size(); ++n) {
      const int tensor_index = op->outputs()->Get(n);
      AllocationInfo* current = &info_[tensor_index];
      if ((current->first_created == -1) || (current->first_created > i)) {
        current->first_created = i;
      }
    }
  }

  // Work out which tensors need to be allocated.
  for (size_t i = 0; i < tensor_count_; ++i) {
    AllocationInfo* current = &info_[i];
    const bool is_read_only =
        (current->first_created == -1) && (current->last_used != -1);
    if (is_read_only) {
      current->needs_allocating = false;
    }
    const bool has_partial_lifetime =
        !is_read_only &&
        ((current->first_created == -1) || (current->last_used == -1));
    if (has_partial_lifetime && current->needs_allocating) {
      TF_LITE_REPORT_ERROR(
          reporter_,
          "Logic error in memory planner, tensor %d has an invalid lifetime: "
          "first_created: %d, last_used: %d",
          i, current->first_created, current->last_used);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus AllocationInfoBuilder::AddScratchBuffers(
    internal::ScratchBufferHandle* buffer_handles) {
  // Set up allocation info for buffers.
  for (size_t i = tensor_count_; i < tensor_count_ + buffer_count_; ++i) {
    AllocationInfo* current = &info_[i];
    internal::ScratchBufferHandle* handle =
        &(buffer_handles[i - tensor_count_]);
    current->output_ptr = reinterpret_cast<void**>(&handle->data);
    current->bytes = handle->bytes;
    current->first_created = handle->node_idx;
    current->last_used = handle->node_idx;
    current->needs_allocating = true;
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace tflite

std::vector<TensorLifetime> GetTensorLifetimes(
    tflite::MicroInterpreter* interpreter) {
  auto error_reporter = interpreter->error_reporter_;
  auto subgraph = interpreter->subgraph_;

  auto numTensors = subgraph->tensors()->size();

  std::vector<uint8_t> buf(1024);
  tflite::SimpleMemoryAllocator allocator(error_reporter, buf.data(),
                                          buf.size());

  tflite::AllocationInfoBuilder builder(error_reporter, &allocator);
  builder.Init(numTensors, interpreter->allocator_.scratch_buffer_count_);
  builder.AddTensors(subgraph, interpreter->context_.tensors);
  builder.AddScratchBuffers(interpreter->allocator_.scratch_buffer_handles_);
  auto allocInfo = builder.Finish();

  std::vector<TensorLifetime> out;
  for (int i = 0; i < numTensors; i++) {
    auto& info = allocInfo[i];
    out.push_back({info.first_created, info.last_used, info.needs_allocating});
  }
  return out;
}
