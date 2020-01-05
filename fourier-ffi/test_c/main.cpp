#include "fourier.h"
#include <array>
#include <complex>
#include <cstdlib>
#include <iostream>

template <typename T> void test() {
  std::array<std::complex<float>, 4> input{{{1, 0}, {0, 0}, {0, 0}, {0, 0}}};
  std::array<std::complex<float>, 4> output;
  fourier::fft<float> fft(input.size());
  fft.transform(input.data(), output.data(), true);
  fft.transform_in_place(output.data(), false);
  for (std::size_t i = 0; i < 4; ++i) {
    if (std::abs(input[i] - output[i]) > 1e-10) {
      std::cerr << "Mismatch at index " << i << " (" << input[i] << " is not "
                << output[i] << ")" << std::endl;
      std::exit(-1);
    }
  }
}

int main() {
  test<float>();
  test<double>();
}
