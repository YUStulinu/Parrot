# Parrot's native library: live audio (miniaudio), speech recognition (whisper.cpp), the phoneme
# model, forced alignment and the pronunciation scorer (onnxruntime), the tutor (llama.cpp) and the
# voice (Piper, onnxruntime). One shared library, `parrot`, with a C API (include/parrot.h) for the
# Flutter app (dart:ffi) and the training code (ctypes).
#
#   cmake -S native -B native/build -DPARROT_ENGINE=ON    (OFF: alignment only, no models)
cmake_minimum_required(VERSION 3.22)
project(parrot_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PARROT_ENGINE "Build the engine (whisper, llama, onnxruntime, audio), not only alignment" ON)
option(PARROT_TESTS "Build the tests and the command-line tool" ON)
option(PARROT_CUDA "Run the language and speech models on an NVIDIA GPU (desktop)" OFF)
set(PARROT_THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party" CACHE PATH "Where scripts/fetch-deps.sh put things")

if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")  # not for nvcc
endif()

# The core: phonemes, the dictionary, alignment. No dependencies but nlohmann/json (from llama.cpp's
# vendor folder).
add_library(parrot_core STATIC src/phones.cpp src/g2p.cpp src/align.cpp)
target_include_directories(parrot_core PUBLIC src include "${PARROT_THIRD_PARTY}/llama.cpp/vendor")

set(PARROT_SOURCES src/api_align.cpp)
if(PARROT_ENGINE)
  # llama.cpp first: it builds ggml, which whisper.cpp then reuses (same ggml, one copy).
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
  set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA ${PARROT_CUDA} CACHE BOOL "" FORCE)
  if(ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
  endif()
  add_subdirectory("${PARROT_THIRD_PARTY}/llama.cpp" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)
  set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  add_subdirectory("${PARROT_THIRD_PARTY}/whisper.cpp" "${CMAKE_BINARY_DIR}/whisper.cpp" EXCLUDE_FROM_ALL)

  # onnxruntime, prebuilt (scripts/fetch-deps.sh).
  if(ANDROID)
    set(ORT_DIR "${PARROT_THIRD_PARTY}/onnxruntime-android")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so"
      INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/headers")
  else()
    set(ORT_DIR "${PARROT_THIRD_PARTY}/onnxruntime-host")
    add_library(onnxruntime SHARED IMPORTED)
    if(WIN32)
      set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_DIR}/lib/onnxruntime.dll" IMPORTED_IMPLIB "${ORT_DIR}/lib/onnxruntime.lib")
    else()
      set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.so")
    endif()
    set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include")
  endif()

  # The engine's parts, as a static library: the shared library and the tests link it.
  add_library(parrot_engine STATIC src/audio.cpp src/asr.cpp src/acoustic.cpp src/scorer.cpp src/tts.cpp
              src/tutor.cpp src/llm.cpp)
  target_include_directories(parrot_engine PRIVATE "${PARROT_THIRD_PARTY}/miniaudio")
  target_link_libraries(parrot_engine PUBLIC parrot_core whisper llama llama-common onnxruntime)
  if(ANDROID)
    target_link_libraries(parrot_engine PUBLIC log OpenSLES aaudio)
  endif()
  set_target_properties(parrot_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
  list(APPEND PARROT_SOURCES src/api.cpp)
endif()

add_library(parrot SHARED ${PARROT_SOURCES})
target_link_libraries(parrot PRIVATE parrot_core)
target_compile_definitions(parrot PRIVATE PARROT_BUILDING)
if(PARROT_ENGINE)
  target_compile_definitions(parrot PRIVATE PARROT_ENGINE)
  target_link_libraries(parrot PRIVATE parrot_engine)
endif()
if(NOT MSVC)
  target_compile_options(parrot PRIVATE -fvisibility=hidden)
endif()

if(PARROT_TESTS AND NOT ANDROID)
  enable_testing()
  add_executable(native_tests tests/native_tests.cpp)
  target_link_libraries(native_tests PRIVATE parrot_core)
  target_compile_definitions(native_tests PRIVATE PARROT_MODELS="${PARROT_THIRD_PARTY}/models")
  add_test(NAME native_tests COMMAND native_tests)
  if(PARROT_ENGINE)
    add_executable(engine_tests tests/engine_tests.cpp)
    target_link_libraries(engine_tests PRIVATE parrot_engine)
    target_compile_definitions(engine_tests PRIVATE PARROT_MODELS="${PARROT_THIRD_PARTY}/models")
    add_test(NAME engine_tests COMMAND engine_tests)
    add_executable(parrot-cli tools/parrot_cli.cpp)
    target_link_libraries(parrot-cli PRIVATE parrot)
    target_include_directories(parrot-cli PRIVATE include "${PARROT_THIRD_PARTY}/llama.cpp/vendor")
    if(WIN32)
      # onnxruntime's DLL next to the programs.
      add_custom_command(TARGET parrot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime.dll" $<TARGET_FILE_DIR:parrot>)
    endif()
  endif()
endif()
