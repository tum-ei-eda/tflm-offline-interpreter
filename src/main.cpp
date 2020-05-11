
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <sstream>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "OfflineOffset.h"
#include "TensorPlanning.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>

std::string GetByteArrayCode(const void *data, size_t len) {
  std::stringstream out;
  out << "{\"";
  for (size_t i = 0; i < len; i++) {
    out << "\\x" << std::setw(2) << std::setfill('0') << std::hex
        << (int)((unsigned char *)data)[i];
  }
  out << "\"}";
  return out.str();
}

std::string FillCodeTemplate(const std::vector<char> &fb, size_t arenaSize,
                             const std::string &setupCode,
                             const std::string &evalCode, int numRegs,
                             int numOps, int numQuants, int intArrayBufSize,
                             int floatArrayBufSize, int inputTensorIndex, int outputTensorIndex)
{
  std::stringstream out;
  out << "// This file is generated. Do not edit.\n";
  {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    out << "// Generated on: " << std::put_time(&tm, "%d.%m.%Y %H:%M:%S")
        << "\n";
  }
  out << R"CODE(
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#ifdef _DEBUG
#include <stdio.h>
#define DBGPRINTF(format, ...) printf(format, ##__VA_ARGS__)
#else
#define DBGPRINTF(format, ...)
#endif

namespace {
)CODE";
  // Flatbuffer. TODO: can drop everything but the actual buffers.
  out << "const unsigned char g_model_data[] __attribute__((aligned(4))) = ";
  out << GetByteArrayCode(fb.data(), fb.size());
  out << ";\n";
  out << "const int g_model_data_len = " << fb.size() << ";\n";
  out << "\n";
  // Tensor buffer size.
  out << "constexpr int kTensorArenaSize = " << arenaSize << ";\n";
  out << "uint8_t tensor_arena[kTensorArenaSize] "
         "__attribute__((aligned(16)));\n";
  out << "\n";
  out << "TfLiteRegistration *g_regOp[" << numRegs << "];\n";
  out << "TfLiteNode g_node[" << numOps << "];\n";
  if (numQuants) {
    out << "TfLiteAffineQuantization g_quants[" << numQuants << "];\n";
  }
  if (floatArrayBufSize) {
    out << "char g_intArrayBuf[" << intArrayBufSize << "];\n";
    out << "char g_floatArrayBuf[" << floatArrayBufSize << "];\n";
  }
  out << R"CODE(TfLiteContext g_ctx{};
} // namespace

void Setup() {
  g_ctx.impl_ = nullptr;
  g_ctx.ReportError = nullptr;
  g_ctx.recommended_num_threads = 1;

  // TODO: CorrectTensorEndianness -> do that offline

)CODE";
  out << setupCode;
  out << R"CODE(}

void *GetInputPtr()
{
  return g_ctx.tensors[)CODE" +
             std::to_string(inputTensorIndex) + R"CODE(].data.data;
}
const void *GetOutputPtr()
{
  return g_ctx.tensors[)CODE" +
             std::to_string(outputTensorIndex) + R"CODE(].data.data;
}

void Eval()
{
)CODE";
  out << evalCode;
  out << R"CODE(}

float SineTestEval(float in)
{
  *(float*)GetInputPtr() = in;
  Eval();
  return *(float*)GetOutputPtr();
}
void TestEval()
{
  Setup();
  auto v1 = SineTestEval(0);
  auto v2 = SineTestEval(3.14f / 2);
  auto v3 = SineTestEval(3.14f);
  auto v4 = SineTestEval((3.14f * 3) / 2);
  auto v5 = SineTestEval(2 * 3.14f);
  DBGPRINTF("0:     %+.02f\n", v1);
  DBGPRINTF("pi/2:  %+.02f\n", v2);
  DBGPRINTF("pi:    %+.02f\n", v3);
  DBGPRINTF("3pi/2: %+.02f\n", v4);
  DBGPRINTF("2pi:   %+.02f\n", v5);
}
)CODE";
  return out.str();
}

// Tracks the last allocation size.
class AllocatorToGetLastAllocSize : public tflite::BuiltinDataAllocator {
public:
  void *Allocate(size_t size, size_t alignment_hint) override {
    lastAllocSize = size;
    return malloc(size);
  }
  void Deallocate(void *data) override { free(data); }
  size_t GetLastAllocSize() { return lastAllocSize; }

private:
  size_t lastAllocSize = 0;
};
size_t GetBuiltinDataSize(tflite::BuiltinOperator opType,
                          const tflite::SubGraph *subgraph) {
  // There seems to be no simple query function for this, so tickle the
  // information out of the parse function.
  auto dummyOp = subgraph->operators()->Get(0);
  tflite::MicroErrorReporter errReporter;
  AllocatorToGetLastAllocSize allocator;
  void *outData;
  tflite::ParseOpData(dummyOp, opType, &errReporter, &allocator, &outData);
  return allocator.GetLastAllocSize();
}

// Aligns a value v to the next value aligned by align bits.
template <typename T> T Align(T v, T align) {
  return (v + align - 1) & ~(align - 1);
}
template <typename T, typename U> T *Align(T *v, U align) {
  return (T *)Align((uintptr_t)v, (uintptr_t)align);
}

static size_t DryRunModelForAllocSize(const tflite::Model *model) {
  std::vector<uint8_t> tmp_arena(100 * 1024);
  tflite::ops::micro::AllOpsResolver resolver;
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroInterpreter interpreter(model, resolver, tmp_arena.data(),
                                       tmp_arena.size(), &error_reporter);
  interpreter.AllocateTensors();

  size_t requiredSize = interpreter.arena_used_bytes();
  for (int i = 0; i < interpreter.tensors_size(); i++) {
    requiredSize +=
        Align(interpreter.tensor(i)->bytes, (size_t)16) + sizeof(TfLiteTensor);
  }

  return requiredSize;
}

static bool Run(const std::string &modelFileName) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter &error_reporter = micro_error_reporter;

  std::ifstream model_file(modelFileName, std::ios::binary | std::ios::ate);
  auto sz = model_file.tellg();
  model_file.seekg(0, std::ios::beg);
  std::vector<char> model_data(sz);
  if (!model_file.read(model_data.data(), sz)) {
    printf("failed to read model file\n");
    return false;
  }

  const tflite::Model *model = tflite::GetModel(model_data.data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter.Report("Model provided is schema version %d not equal "
                          "to supported version %d.",
                          model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  auto tensorArenaSize = DryRunModelForAllocSize(model);
  std::vector<uint8_t> tensorArena(tensorArenaSize + 16);
  uint8_t *tensor_arena = Align(tensorArena.data(), 16);

  OfflineOffset::Init(tensor_arena, tensorArenaSize, model_data);

  auto subgraphs = model->subgraphs();
  if (subgraphs->size() != 1) {
    printf("Only 1 subgraph supported\n");
    return false;
  }
  auto subgraph = (*subgraphs)[0];
  auto tensors = subgraph->tensors();
  auto operators = subgraph->operators();
  auto inputTensorIndex = subgraph->inputs()->Get(0);
  auto outputTensorIndex = subgraph->outputs()->Get(0);

  tflite::ops::micro::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensorArenaSize, &error_reporter);

  // Must be called before AllocateTensors.
  auto lifetimes = GetTensorLifetimes(&interpreter);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter.Report("AllocateTensors() failed");
    return false;
  }

  std::vector<uint8_t> plannerBuf(1024);
  tflite::GreedyMemoryPlanner planner(plannerBuf.data(), plannerBuf.size());
  printf("num tensors: %lu\n", interpreter.tensors_size());
  std::map<int, int> tensorToPlanBuffer;
  for (int i = 0; i < interpreter.tensors_size(); i++) {
    if (!lifetimes[i].needsAlloc)
      continue;
    size_t sz = Align(interpreter.tensor(i)->bytes, (size_t)16);
    planner.AddBuffer(&error_reporter, sz, lifetimes[i].firstUse,
                      lifetimes[i].lastUse);
    tensorToPlanBuffer[i] = planner.GetBufferCount() - 1;
  }
  planner.PrintMemoryPlan(&error_reporter);

  std::stringstream setupCode;
  int numQuants = 0;
  int intArrayBufSize = 0;
  int floatArrayBufSize = 0;

  setupCode << "  // Setup tensors.\n";
  setupCode << "  g_ctx.tensors_size = " << tensors->size() << ";\n";
  setupCode << "  g_ctx.tensors = (TfLiteTensor*)"
            << OfflineOffset(tensor_arena + planner.GetMaximumMemorySize())
                   .getPtrCode()
            << ";\n";
  setupCode << "  for (size_t i = 0; i < " << interpreter.tensors_size() << "; i++) {\n";
  setupCode << "    TfLiteTensor *tensor = &g_ctx.tensors[i];\n";
  setupCode << "    *tensor = {};\n";
  setupCode << "    //ConvertTensorType();\n";
  setupCode << "  }\n";
  for (int i = 0; i < interpreter.tensors_size(); i++) {
    OfflineOffset tensorDataOffset(interpreter.tensor(i)->data.data);
    if (lifetimes[i].needsAlloc) {
      int bufferOffset = 0;
      planner.GetOffsetForBuffer(&error_reporter, tensorToPlanBuffer[i],
                                 &bufferOffset);
      tensorDataOffset.set(tensor_arena + bufferOffset);
    }

    std::string tensorI = "  g_ctx.tensors[" + std::to_string(i) + "]";
    setupCode << tensorI << ".data.data = (void*)"
              << tensorDataOffset.getPtrCode() << ";\n";
    // TODO: Do these assignments offline. Tricky: ABI differences
    TfLiteType type;
    ConvertTensorType(tensors->Get(i)->type(), &type, &error_reporter);
    setupCode << tensorI << ".type = (TfLiteType)" << type << "; // " << TfLiteTypeGetName(type) << "\n";
    setupCode << tensorI << ".is_variable = " << tensors->Get(i)->is_variable()
              << ";\n";
    setupCode << tensorI << ".allocation_type = "
              << ((tensorDataOffset.getType() == OfflineOffset::Type::FB)
                      ? "kTfLiteMmapRo"
                      : "kTfLiteArenaRw")
              << ";\n";
    setupCode << tensorI << ".bytes = " << interpreter.tensor(i)->bytes << ";\n";
    setupCode << tensorI << ".dims = (TfLiteIntArray*)"
              << OfflineOffset(tensors->Get(i)->shape()).getPtrCode() << ";\n";
    auto quant = tensors->Get(i)->quantization();
    if (quant && quant->scale() && quant->scale()->size() > 0 &&
        quant->zero_point() && quant->zero_point()->size() > 0) {
      printf("tensor has quantization!\n");

      setupCode << tensorI << ".params.scale = " << quant->scale()->Get(0)
                << ";\n";
      setupCode << tensorI << ".params.zero_point = "
                << (int32_t)quant->zero_point()->Get(0) << ";\n";

      std::string quantI = "  g_quants[" + std::to_string(numQuants) + "]";
      numQuants++;
      int channels = quant->scale()->size();
      setupCode << quantI << ".zero_point = (TfLiteIntArray*)&g_intArrayBuf[" << intArrayBufSize << "];\n";
      intArrayBufSize += TfLiteIntArrayGetSizeInBytes(channels);
      setupCode << quantI << ".scale = (TfLiteFloatArray*)&g_floatArrayBuf[" << floatArrayBufSize << "];\n";
      floatArrayBufSize += TfLiteFloatArrayGetSizeInBytes(channels);
      setupCode << quantI << ".zero_point->size = " << channels << ";\n";
      setupCode << quantI << ".scale->size = " << channels << ";\n";
      for (int c = 0; c < channels; c++) {
        setupCode << quantI << ".zero_point->data[" << c
                  << "] = " << quant->zero_point()->Get(c) << ";\n";
        setupCode << quantI << ".scale->data[" << c
                  << "] = " << quant->scale()->Get(c) << ";\n";
      }
      setupCode << quantI
                << ".quantized_dimension = " << quant->quantized_dimension()
                << ";\n";
      setupCode << tensorI << ".quantization = {kTfLiteAffineQuantization, &"
                << quantI << "};\n";
    }
    // Do not copy tensor name, not used on target.
  }
  setupCode << "\n";

  struct Op {
    tflite::BuiltinOperator code;
    int version;
    bool operator<(const Op &op) const {
      if (code == op.code)
        return version < op.version;
      return code < op.code;
    }
    bool operator==(const Op &op) {
      return code == op.code && version == op.version;
    }
  };
  std::vector<Op> usedRegistrations;
  std::vector<int> opToRegistration;
  auto nOps = interpreter.operators_size();
  for (int i = 0; i < nOps; i++) {
    auto nodeAndReg = interpreter.node_and_registration(i);
    auto node = &nodeAndReg.node;
    auto reg = nodeAndReg.registration;
    auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    auto GetFBOffset = [&](void *p) {
      return (void *)((uintptr_t)p - (uintptr_t)model_data.data());
    };

    printf("operation %i: %s\n", i, tflite::EnumNamesBuiltinOperator()[code]);

    Op op{code, reg->version};
    auto itOp =
        std::find(usedRegistrations.begin(), usedRegistrations.end(), op);
    if (itOp == usedRegistrations.end()) {
      itOp = usedRegistrations.insert(usedRegistrations.end(), op);
    }
    opToRegistration.push_back(itOp - usedRegistrations.begin());

    // Build node.
    setupCode << "  {\n";
    setupCode << "    TfLiteNode &node = g_node[" << i << "];\n";
    setupCode << "    node.inputs = (TfLiteIntArray*)"
              << OfflineOffset(node->inputs).getPtrCode() << ";\n";
    setupCode << "    node.outputs = (TfLiteIntArray*)"
              << OfflineOffset(node->outputs).getPtrCode() << ";\n";
    setupCode << "    node.temporaries = nullptr;\n";
    setupCode << "    node.user_data = nullptr;\n";
    // TODO: ABI incompatibilities?
    setupCode << "    static char builtin_data[] = "
              << GetByteArrayCode(node->builtin_data,
                                  GetBuiltinDataSize(code, subgraph))
              << ";\n";
    setupCode << "    node.builtin_data = builtin_data;\n";
    setupCode << "    node.custom_initial_data = "
              << OfflineOffset(node->custom_initial_data).getPtrCode() << ";\n";
    setupCode << "    node.custom_initial_data_size = "
              << node->custom_initial_data_size << ";\n";
    setupCode << "    node.delegate = nullptr;\n";
    setupCode << "  }\n";
  }

  {
    int i = 0;
    for (const auto &reg : usedRegistrations) {
      auto opName = tflite::EnumNameBuiltinOperator(reg.code);
      setupCode << "  g_regOp[" << i << "] = tflite::ops::micro::Register_"
                << opName << "();\n";
      i++;
    }
  }

  std::stringstream evalCode;
  for (int i = 0; i < nOps; i++) {
    // init and free are not implemented, so don't call them.
    if (interpreter.node_and_registration(i).registration->prepare) {
      evalCode << "  g_regOp[" << opToRegistration[i]
               << "]->prepare(&g_ctx, &g_node[" << i << "]);\n";
    }
    evalCode << "  g_regOp[" << opToRegistration[i]
             << "]->invoke(&g_ctx, &g_node[" << i << "]);\n";
  }

  // TODO: ABI incompatible
  size_t totalTensorBufSize =
      planner.GetMaximumMemorySize() + interpreter.tensors_size() * sizeof(TfLiteTensor);

  std::ofstream outFile("../../../out/tflm_src/target_src.cc");
  outFile << FillCodeTemplate(model_data, totalTensorBufSize, setupCode.str(),
                              evalCode.str(), usedRegistrations.size(), nOps,
                              numQuants, intArrayBufSize, floatArrayBufSize, inputTensorIndex,
                              outputTensorIndex);

  printf("Required tensor memory: %lu\n", totalTensorBufSize);

  auto Test = [&](float x_val) {
    interpreter.input(0)->data.f[0] = x_val;
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter.Report("Invoke failed on x_val: %f\n",
                            static_cast<double>(x_val));
      return -1.0f;
    }
    return interpreter.output(0)->data.f[0];
  };

  printf("0:     %+.02f\n", Test(0));
  printf("pi/2:  %+.02f\n", Test(3.14f / 2));
  printf("pi:    %+.02f\n", Test(3.14f));
  printf("3pi/2: %+.02f\n", Test((3 * 3.14f) / 2));
  printf("2pi:   %+.02f\n", Test(2 * 3.14f));

  return true;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("usage: %s modelFile.tflite\n", argv[0]);
    return 1;
  }

  if (!Run(argv[1])) {
    return 1;
  }

  return 0;
}
