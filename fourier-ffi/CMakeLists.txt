cmake_minimum_required(VERSION 3.7)
project(fourier)

enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

if (MSVC)
    add_compile_options(/W4 /WX)
else ()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/cargo)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build --target-dir=${CARGO_TARGET_DIR})
    set(CARGO_BINARY_DIR "${CARGO_TARGET_DIR}/debug")
else ()
    set(ENV{RUSTFLAGS} "-C link-arg=-s")
    set(CARGO_CMD cargo build --release --target-dir=${CARGO_TARGET_DIR})
    set(CARGO_BINARY_DIR "${CARGO_TARGET_DIR}/release")
endif ()

set(FOURIER_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}fourier${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(FOURIER_DLL "${CMAKE_SHARED_LIBRARY_PREFIX}fourier${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(FOURIER_DLL_IMP "${FOURIER_DLL}${CMAKE_STATIC_LIBRARY_SUFFIX}")
if (WIN32)
    set(CARGO_BINARIES ${CARGO_BINARY_DIR}/${FOURIER_LIB} ${CARGO_BINARY_DIR}/${FOURIER_DLL} ${CARGO_BINARY_DIR}/${FOURIER_DLL_IMP})
else ()
    set(CARGO_BINARIES ${CARGO_BINARY_DIR}/${FOURIER_LIB} ${CARGO_BINARY_DIR}/${FOURIER_DLL})
endif ()

add_custom_target(fourier_cargo
    COMMAND ${CARGO_CMD}
    COMMAND ${CMAKE_COMMAND} -E copy ${CARGO_BINARIES} ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads)

add_library(${FOURIER_LIB} STATIC IMPORTED)
add_dependencies(${FOURIER_LIB} fourier_cargo)
target_link_libraries(${FOURIER_LIB} INTERFACE ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${FOURIER_LIB} PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_LIB})
set_property(TARGET ${FOURIER_LIB} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(${FOURIER_DLL} SHARED IMPORTED)
add_dependencies(${FOURIER_DLL} fourier_cargo)
set_property(TARGET ${FOURIER_DLL} PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_DLL})
set_property(TARGET ${FOURIER_DLL} PROPERTY IMPORTED_IMPLIB ${CMAKE_CURRENT_BINARY_DIR}/${FOURIER_DLL_IMP})
set_property(TARGET ${FOURIER_DLL} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_c")

# Library orderings are magic and this is the correct incantation.
if (NOT MSVC)
    include(CheckLibraryExists)
    check_library_exists(m cabsf "" HAS_LIBM)
    if (HAS_LIBM)
        set(LIBM "m")
    else ()
        set(LIBM "")
    endif ()

    add_executable(test_c_static ${CMAKE_CURRENT_SOURCE_DIR}/test.c)
    target_link_libraries(test_c_static ${LIBM} ${FOURIER_LIB})
    add_test(c_static test_c_static)

    add_executable(test_c_dynamic ${CMAKE_CURRENT_SOURCE_DIR}/test.c)
    target_link_libraries(test_c_dynamic ${LIBM} ${FOURIER_DLL})
    add_test(c_dynamic test_c_dynamic)
endif ()

add_executable(test_cpp_static ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(test_cpp_static ${FOURIER_LIB})
add_test(cpp_static test_cpp_static)

add_executable(test_cpp_dynamic ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(test_cpp_dynamic ${FOURIER_DLL})
add_test(cpp_dynamic test_cpp_dynamic)
