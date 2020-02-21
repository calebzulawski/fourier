# Fourier C/C++ interface

To build the library using CMake:
```bash
# Configure
cmake -S . -B build

# Build
cmake --build build

# Test
cd build && ctest
```

Uses the default Rust toolchain, so remember to change it (`rustup default <toolchain>`) if compiling for a non-default target.
