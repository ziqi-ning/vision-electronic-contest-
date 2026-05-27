cmake_minimum_required(VERSION 3.20)
project(FVS-Cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Export compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ——————————————————————————————————
# Options
option(BUILD_TESTS "Build unit and integration tests" ON)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UBSanitizer in Debug" ON)

# ——————————————————————————————————
# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(ENABLE_SANITIZERS)
        if(MSVC)
            # MSVC sanitizers not yet supported via flags; rely on Debug heap
        else()
            add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
            add_link_options(-fsanitize=address,undefined)
        endif()
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        add_compile_options(-O3)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
            include(CheckIPOSupported)
            include(ExternalProject)
            add_compile_options(-flto)
            add_link_options(-flto)
        endif()
    endif()
endif()

# ——————————————————————————————————
# Conan dependencies (optional — only if conanbuildinfo.cmake exists)
if(EXISTS "${CMAKE_BINARY_DIR}/conanbuildinfo.cmake")
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
endif()

# MSYS2 / standalone dependency paths (fallback)
set(MSYS2_PREFIX "C:/msys64/mingw64")
if(NOT DEFINED OpenCV_DIR AND EXISTS "${MSYS2_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${MSYS2_PREFIX}")
endif()

# Find packages
find_package(OpenCV 4.8 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

# ——————————————————————————————————
# Unity Build — group source files
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)

# ——————————————————————————————————
# Public include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
if(DEFINED CONAN_INCLUDE_DIRS)
    include_directories(${CONAN_INCLUDE_DIRS})
endif()
# MSYS2 fallback
if(NOT DEFINED CONAN_INCLUDE_DIRS)
    include_directories("${MSYS2_PREFIX}/include")
endif()

# ——————————————————————————————————
# Source groups
file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# ——————————————————————————————————
# Main executable
add_executable(fvs_main
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(fvs_main
    PUBLIC
        ${OpenCV_LIBS}
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        ${CONAN_LIBS}
)

# MSYS2 fallback link libs (when Conan not used)
if(NOT DEFINED CONAN_LIBS)
    target_link_directories(fvs_main PRIVATE "${MSYS2_PREFIX}/lib")
endif()

target_include_directories(fvs_main
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# ——————————————————————————————————
# Unit tests (GoogleTest)
if(BUILD_TESTS)
    enable_testing()

    find_package(GTest QUIET)

    if(TARGET GTest::GTest)
        message(STATUS "GoogleTest found (system or conan)")
    else()
        message(STATUS "GoogleTest not found — tests will not be built")
    endif()

    if(TARGET GTest::GTest)
        file(GLOB_RECURSE UNIT_TEST_SOURCES
            "${CMAKE_SOURCE_DIR}/tests/unit/*.cpp"
        )

        add_executable(fvs_unit_tests ${UNIT_TEST_SOURCES})

        target_link_libraries(fvs_unit_tests
            PRIVATE
                GTest::GTest
                GTest::GTestMain
                ${OpenCV_LIBS}
                spdlog::spdlog
                yaml-cpp::yaml-cpp
        )

        target_include_directories(fvs_unit_tests
            PRIVATE
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/src
        )

        include(GoogleTest)
        gtest_discover_tests(fvs_unit_tests)
    endif()
endif()

# ——————————————————————————————————
# clang-tidy integration (opt-in via cmake -DENABLE_CLANG_TIDY=ON)
option(ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)
if(ENABLE_CLANG_TIDY AND NOT MSVC)
    find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy)
    if(CLANG_TIDY_EXECUTABLE)
        message(STATUS "clang-tidy enabled")
        set_target_properties(fvs_main PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
        if(BUILD_TESTS AND TARGET fvs_unit_tests)
            set_target_properties(fvs_unit_tests PROPERTIES
                CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
        endif()
    else()
        message(WARNING "clang-tidy not found — skipping")
    endif()
endif()

# ——————————————————————————————————
# Install targets
install(TARGETS fvs_main RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "=== FVS-Cpp Build Configuration ===")
message(STATUS "  CMake Version : ${CMAKE_VERSION}")
message(STATUS "  C++ Standard  : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  Source dir    : ${CMAKE_SOURCE_DIR}")
message(STATUS "  Build dir     : ${CMAKE_BINARY_DIR}")
message(STATUS "==================================")
