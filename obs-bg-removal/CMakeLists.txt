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

target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.c src/bg-filter.cpp)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
