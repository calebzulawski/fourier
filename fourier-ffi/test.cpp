#include "fourier.h"
#include <array>
#include <complex>
#include <cstdlib>
#include <iostream>

template <typename C> void check(const C &input, const C &output) {
  for (std::size_t i = 0; i < 4; ++i) {
    if (std::abs(input[i] - output[i]) > 1e-10) {
      std::cerr << "Mismatch at index " << i << " (" << input[i] << " is not "
                << output[i] << ")" << std::endl;
      std::exit(-1);
    }
  }
}

template <typename T> void test() {
  std::array<std::complex<float>, 4> input{{{1, 0}, {0, 0}, {0, 0}, {0, 0}}};
  std::array<std::complex<float>, 4> output;
  fourier::fft<float> fft(input.size());
  fft.transform(input.data(), output.data(), fourier::transform::fft);
  fft.transform_in_place(output.data(), fourier::transform::ifft);
  check(input, output);
}

void test_c_float() {
  using namespace fourier::c;
  std::array<std::complex<float>, 4> input{{{1, 0}, {0, 0}, {0, 0}, {0, 0}}};
  std::array<std::complex<float>, 4> output;
  fourier_fft_float *fft = fourier_create_float(4);
  fourier_transform_float(fft, input.data(), output.data(),
                          FOURIER_TRANSFORM_FFT);
  fourier_transform_in_place_float(fft, output.data(), FOURIER_TRANSFORM_IFFT);
  fourier_destroy_float(fft);
  check(input, output);
}

void test_c_double() {
  using namespace fourier::c;
  std::array<std::complex<double>, 4> input{{{1, 0}, {0, 0}, {0, 0}, {0, 0}}};
  std::array<std::complex<double>, 4> output;
  fourier_fft_double *fft = fourier_create_double(4);
  fourier_transform_double(fft, input.data(), output.data(),
                           FOURIER_TRANSFORM_FFT);
  fourier_transform_in_place_double(fft, output.data(), FOURIER_TRANSFORM_IFFT);
  fourier_destroy_double(fft);
  check(input, output);
}

int main() {
  test<float>();
  test<double>();
  test_c_float();
  test_c_double();
  std::cout << "Tests ran successfully." << std::endl;
  return 0;
}
