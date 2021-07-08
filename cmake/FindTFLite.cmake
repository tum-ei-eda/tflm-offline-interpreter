IF(NOT TF_SRC)
    INCLUDE(FetchContent)
    IF(TF_TAG)
        MESSAGE(STATUS "Getting TF tag '${TF_TAG}' and not master")
        FetchContent_Declare(
            tf 
            GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
            GIT_PROGRESS FALSE
            GIT_TAG ${TF_TAG}
            QUIET
            )
    ELSE()
        FetchContent_Declare(
            tf 
            GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
            GIT_PROGRESS FALSE
            QUIET
            )
    ENDIF()
    FetchContent_GetProperties(tf)
    IF(NOT tf_POPULATED)
        MESSAGE(STATUS "TensorFlow sources not given/populated, fetching from GH...")
        FetchContent_Populate(tf)
    ENDIF()
    SET(TF_SRC ${tf_SOURCE_DIR})

    FetchContent_Declare(
        flatbuffers 
        GIT_REPOSITORY https://github.com/google/flatbuffers.git 
        GIT_PROGRESS FALSE 
        QUIET
        )
    FetchContent_GetProperties(flatbuffers)
    IF(NOT flatbuffers_POPULATED)
        MESSAGE(STATUS "Now getting 'flatbuffers'...")
        FetchContent_Populate(flatbuffers)
    ENDIF()
    LIST(APPEND TFL_INC_DIRS ${flatbuffers_SOURCE_DIR}/include)

    FetchContent_Declare(
        fixedpoint 
        GIT_REPOSITORY https://github.com/google/gemmlowp.git 
        GIT_PROGRESS FALSE 
        QUIET 
        )
    FetchContent_GetProperties(fixedpoint)
    IF(NOT fixedpoint_POPULATED)
        MESSAGE(STATUS "And finaly 'fixedpoint'...")
        FetchContent_Populate(fixedpoint)
    ENDIF()
    LIST(APPEND TFL_INC_DIRS ${fixedpoint_SOURCE_DIR})

    FetchContent_Declare(
        ruy 
        GIT_REPOSITORY https://github.com/google/ruy.git 
        GIT_PROGRESS FALSE 
        QUIET 
        )
    FetchContent_GetProperties(ruy)
    IF(NOT ruy_POPULATED)
        MESSAGE(STATUS "Oh we also need 'ruy'...")
        FetchContent_Populate(ruy)
    ENDIF()
    LIST(APPEND TFL_INC_DIRS ${ruy_SOURCE_DIR})
ENDIF()

SET(TFL_SRC ${TF_SRC}/tensorflow/lite)
SET(TFLM_SRC ${TFL_SRC}/micro)
SET(TFLD_SRC ${TFL_SRC}/tools/make/downloads)

IF(EXISTS ${TFLD_SRC}/flatbuffers/include)
    LIST(APPEND TFL_INC_DIRS ${TFLD_SRC}/flatbuffers/include)
ENDIF()

IF(EXISTS ${TFLD_SRC}/gemmlowp)
    LIST(APPEND ${TFLD_SRC}/gemmlowp)
ENDIF()

LIST(APPEND TFL_INC_DIRS 
    ${TF_SRC}
    )

SET(TFL_SRCS
    # Not really needed?
    ${TFLM_SRC}/micro_error_reporter.cc
    ${TFLM_SRC}/debug_log.cc
    ${TFLM_SRC}/micro_string.cc

    # For reporter->Report
    ${TF_SRC}/tensorflow/lite/core/api/error_reporter.cc

    # Kernels
    ${TFLM_SRC}/kernels/all_ops_resolver.cc
    ${TFLM_SRC}/kernels/depthwise_conv.cc
    ${TFLM_SRC}/kernels/softmax.cc
    ${TFLM_SRC}/kernels/fully_connected.cc
    ${TFLM_SRC}/kernels/depthwise_conv.cc
    ${TFLM_SRC}/kernels/pooling.cc
    ${TFLM_SRC}/kernels/logical.cc
    ${TFLM_SRC}/kernels/logistic.cc
    ${TFLM_SRC}/kernels/svdf.cc
    ${TFLM_SRC}/kernels/concatenation.cc
    ${TFLM_SRC}/kernels/ceil.cc
    ${TFLM_SRC}/kernels/floor.cc
    ${TFLM_SRC}/kernels/prelu.cc
    ${TFLM_SRC}/kernels/neg.cc
    ${TFLM_SRC}/kernels/elementwise.cc
    ${TFLM_SRC}/kernels/maximum_minimum.cc
    ${TFLM_SRC}/kernels/arg_min_max.cc
    ${TFLM_SRC}/kernels/reshape.cc
    ${TFLM_SRC}/kernels/comparisons.cc
    ${TFLM_SRC}/kernels/round.cc
    ${TFLM_SRC}/kernels/strided_slice.cc
    ${TFLM_SRC}/kernels/pack.cc
    ${TFLM_SRC}/kernels/pad.cc
    ${TFLM_SRC}/kernels/split.cc
    ${TFLM_SRC}/kernels/add.cc
    ${TFLM_SRC}/kernels/mul.cc
    ${TFLM_SRC}/kernels/unpack.cc
    ${TFLM_SRC}/kernels/quantize.cc
    ${TFLM_SRC}/kernels/activations.cc
    ${TFLM_SRC}/kernels/dequantize.cc
    ${TFLM_SRC}/kernels/conv.cc
    ${TFLM_SRC}/kernels/reduce.cc
    # Kernel deps
    ${TFLM_SRC}/micro_utils.cc
    ${TFL_SRC}/kernels/internal/quantization_util.cc
    ${TFL_SRC}/kernels/kernel_util.cc

    ${TFL_SRC}/c/common.c

    ${TFLM_SRC}/micro_interpreter.cc
    ${TFLM_SRC}/micro_allocator.cc
    ${TFLM_SRC}/simple_memory_allocator.cc
    ${TFLM_SRC}/memory_helpers.cc
    ${TFLM_SRC}/memory_planner/greedy_memory_planner.cc
    ${TFL_SRC}/core/api/tensor_utils.cc
    ${TFL_SRC}/core/api/flatbuffer_conversions.cc
    ${TFL_SRC}/core/api/op_resolver.cc
    )

ADD_LIBRARY(tflite STATIC
    ${TFL_SRCS}
)

TARGET_INCLUDE_DIRECTORIES(tflite PUBLIC
    ${TFL_INC_DIRS}
)

TARGET_COMPILE_DEFINITIONS(tflite PUBLIC
    TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
    TF_LITE_STATIC_MEMORY
    TFLITE_EMULATE_FLOAT
    "$<$<CONFIG:RELEASE>:TF_LITE_STRIP_ERROR_STRINGS>"
)

SET(TFLite_INCLUDE_DIRS 
    ${TFL_INC_DIRS}
    )

SET(TFLite_SOURCES 
    ${TFL_SRCS}
    )

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(TFLite DEFAULT_MSG TFLite_INCLUDE_DIRS TFLite_SOURCES)
