cmake_minimum_required(VERSION 3.7)
project(fourier)

enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build)
    set(TARGET_DIR "debug")
else ()
    set(ENV{RUSTFLAGS} "-C link-arg=-s")
    set(CARGO_CMD cargo build --release)
    set(TARGET_DIR "release")
endif ()

set(FOURIER_DLL "${CMAKE_SHARED_LIBRARY_PREFIX}fourier${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(FOURIER_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}fourier${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_custom_target(fourier_cargo
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} ${CARGO_CMD} 
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/${FOURIER_DLL}" ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/${FOURIER_LIB}" ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(${FOURIER_LIB} STATIC IMPORTED)
add_dependencies(${FOURIER_LIB} fourier_cargo)
set_property(TARGET ${FOURIER_LIB} PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_LIB})

add_library(${FOURIER_DLL} SHARED IMPORTED)
add_dependencies(${FOURIER_DLL} fourier_cargo)
set_property(TARGET ${FOURIER_DLL} PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_DLL})

if (UNIX)
    set(TAR_SUFFIX .tar.gz)
    set(TAR_FLAGS cfvz)
    set(TAR_TYPE gnutar)
else ()
    set(TAR_SUFFIX .zip)
    set(TAR_FLAGS cfv)
    set(TAR_TYPE zip)
endif ()
add_custom_target(package
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/include/fourier.h ${CMAKE_CURRENT_BINARY_DIR}/fourier/include/fourier.h
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_DLL} ${CMAKE_CURRENT_BINARY_DIR}/fourier/lib/${FOURIER_DLL}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_LIB} ${CMAKE_CURRENT_BINARY_DIR}/fourier/lib/${FOURIER_LIB}
    COMMAND ${CMAKE_COMMAND} -E tar ${TAR_FLAGS} "fourier${TAR_SUFFIX}" --format=${TAR_TYPE}
        ${CMAKE_CURRENT_BINARY_DIR}/fourier/lib/${FOURIER_DLL}
        ${CMAKE_CURRENT_BINARY_DIR}/fourier/lib/${FOURIER_LIB}
        ${CMAKE_CURRENT_BINARY_DIR}/fourier/include/fourier.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${FOURIER_DLL} ${FOURIER_LIB})

set(CSOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_c")
set(CINCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads)
set(STATIC_DEPS "${CMAKE_DL_LIBS}" "${CMAKE_THREAD_LIBS_INIT}")

include(CheckLibraryExists)
check_library_exists(m cabsf "" HAS_LIBM)
if (HAS_LIBM)
    set(LIBM "m")
else ()
    set(LIBM "")
endif ()

add_executable(test_c_static "${CSOURCE}/main.c")
target_link_libraries(test_c_static ${LIBM} ${STATIC_DEPS} ${FOURIER_LIB})
target_include_directories(test_c_static PRIVATE ${CINCLUDE})

add_executable(test_c_dynamic "${CSOURCE}/main.c")
target_link_libraries(test_c_dynamic ${LIBM} ${FOURIER_DLL})
target_include_directories(test_c_dynamic PRIVATE ${CINCLUDE})

add_executable(test_cpp_static "${CSOURCE}/main.cpp")
target_link_libraries(test_cpp_static ${STATIC_DEPS} ${FOURIER_LIB})
target_include_directories(test_cpp_static PRIVATE ${CINCLUDE})

add_executable(test_cpp_dynamic "${CSOURCE}/main.cpp")
target_link_libraries(test_cpp_dynamic ${FOURIER_DLL})
target_include_directories(test_cpp_dynamic PRIVATE ${CINCLUDE})

add_test(c_static test_c_static)
add_test(c_dynamic test_c_dynamic)
add_test(cpp_static test_cpp_static)
add_test(cpp_dynamic test_cpp_dynamic)
