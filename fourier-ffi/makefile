.PHONfY: clean test

RELEASE_PATH = ../target/release
WARNING_FLAGS = -Wall -Wextra -Wpedantic -Werror
FOURIER_FLAGS = -I include -L ${RELEASE_PATH} -lfourier

ifneq ($(OS),Windows_NT)
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
    endif
    ifeq ($(UNAME_S),Darwin)
        STRIP = strip -x ${RELEASE_PATH}/libfourier.dylib
    endif
endif

${RELEASE_PATH}:
	cargo build --release
ifdef STRIP
	${STRIP}
endif

test_bin:
	mkdir -p test_bin

clean:
	cargo clean
	rm -rf test_bin

test: ../target/release test_bin
	${CXX} -g -std=c++11 ${WARNING_FLAGS} ${FOURIER_FLAGS} test_c/main.cpp -o test_bin/test_cpp_dynamic
	${CC} -g -std=c99 ${WARNING_FLAGS} ${FOURIER_FLAGS} test_c/main.c -o test_bin/test_c_dynamic
	test_bin/test_cpp_dynamic
	test_bin/test_c_dynamic
