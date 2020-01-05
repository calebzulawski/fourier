#include "fourier.h"
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void test_float() {
  float complex input[4] = {1, 0, 0, 0};
  float complex output[4];
  struct fourier_fft_float *fft = fourier_create_float(4);
  fourier_transform_float(fft, input, output, true);
  fourier_transform_in_place_float(fft, output, false);
  fourier_destroy_float(fft);
  for (int i = 0; i < 4; i++) {
    if (cabsf(input[i] - output[i]) > 1e-10f) {
      fprintf(stderr, "Mismatch at index %d (%f%+fi is not %f%+fi)", i,
              crealf(input[i]), cimagf(input[i]), crealf(output[i]),
              cimagf(output[i]));
      exit(-1);
    }
  }
}

void test_double() {
  double complex input[4] = {1, 0, 0, 0};
  double complex output[4];
  struct fourier_fft_double *fft = fourier_create_double(4);
  fourier_transform_double(fft, input, output, true);
  fourier_transform_in_place_double(fft, output, false);
  fourier_destroy_double(fft);
  for (int i = 0; i < 4; i++) {
    if (cabs(input[i] - output[i]) > 1e-10f) {
      fprintf(stderr, "Mismatch at index %d (%f%+fi is not %f%+fi)", i,
              creal(input[i]), cimag(input[i]), creal(output[i]),
              cimag(output[i]));
      exit(-1);
    }
  }
}

int main() {
  test_float();
  test_double();
}
