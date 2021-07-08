CMAKE_MINIMUM_REQUIRED(VERSION 3.13 FATAL_ERROR)
PROJECT(tflm-offline-interpreter)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(TFLite REQUIRED)

ADD_EXECUTABLE(${PROJECT_NAME}
    src/main.cpp
    src/OfflineOffset.cpp
    src/TensorPlanning.cpp
    src/OptimalMemPlanner.cpp
    src/MemMap.cpp
)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC tflite)
