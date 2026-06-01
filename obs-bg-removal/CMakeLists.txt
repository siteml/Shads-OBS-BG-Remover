cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 20)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>)
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES
      AUTOMOC ON
      AUTOUIC ON
      AUTORCC ON)
endif()

# ONNX Runtime (DirectML NuGet build)
# Defaults to a repo-local deps/onnxruntime folder (git-ignored via deps/).
# Override with -DONNXRUNTIME_DIR=<path> if your ONNX Runtime lives elsewhere.
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime" CACHE PATH "Path to ONNX Runtime")

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# Copy ONNX Runtime DLLs to output folder
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>
)

# OpenVINO (native runtime) - powers the Intel Arc / GPU inference backend.
# Optional: if deps/openvino is present, the OpenVINO backend is compiled in;
# otherwise the plugin still builds with just the ONNX Runtime backend.
# Override with -DOpenVINO_DIR=<path>/runtime/cmake to use another copy.
set(OpenVINO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/openvino/runtime/cmake"
    CACHE PATH "Path to OpenVINO CMake config")
# OpenVINO's config pulls in its bundled oneTBB; point CMake at the archive copy.
set(TBB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/openvino/runtime/3rdparty/tbb/lib/cmake/TBB"
    CACHE PATH "Path to OpenVINO's bundled TBB")
# A RelWithDebInfo build should consume OpenVINO's Release import libs.
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO "RelWithDebInfo;Release;")

find_package(OpenVINO QUIET COMPONENTS Runtime)
if(OpenVINO_FOUND)
  message(STATUS "OpenVINO found - building the OpenVINO inference backend")
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE openvino::runtime)
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_OPENVINO=1)

  # Copy OpenVINO runtime, plugins, frontends and TBB next to the plugin DLL
  # so the loader finds them when OBS loads us.
  set(_OV_BIN "${CMAKE_CURRENT_SOURCE_DIR}/deps/openvino/runtime/bin/intel64/Release")
  set(_OV_TBB "${CMAKE_CURRENT_SOURCE_DIR}/deps/openvino/runtime/3rdparty/tbb/bin")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_OV_BIN}/openvino.dll"
          "${_OV_BIN}/openvino_intel_gpu_plugin.dll"
          "${_OV_BIN}/openvino_intel_cpu_plugin.dll"
          "${_OV_BIN}/openvino_onnx_frontend.dll"
          "${_OV_BIN}/openvino_ir_frontend.dll"
          "${_OV_TBB}/tbb12.dll"
          "${_OV_TBB}/tbbbind_2_5.dll"
          "${_OV_TBB}/tbbmalloc.dll"
          "${_OV_TBB}/tbbmalloc_proxy.dll"
          $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>
  )
else()
  message(STATUS "OpenVINO not found at ${OpenVINO_DIR} - building without the OpenVINO backend")
endif()

target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.c src/bg-filter.cpp)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
