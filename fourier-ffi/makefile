.PHONfY: clean test

WARNING_FLAGS = -Wall -Wextra -Wpedantic -Werror
FOURIER_FLAGS = -I include -L target/release -lfourier

test_bin:
	mkdir -p test_bin

target/release:
	cargo build --release

clean:
	cargo clean
	rm -rf test_bin

test: target/release test_bin
	${CXX} -g -std=c++11 ${WARNING_FLAGS} ${FOURIER_FLAGS} test_c/main.cpp -o test_bin/test_cpp_dynamic
	${CC} -g -std=c99 ${WARNING_FLAGS} ${FOURIER_FLAGS} test_c/main.c -o test_bin/test_c_dynamic
	test_bin/test_cpp_dynamic
	test_bin/test_c_dynamic
