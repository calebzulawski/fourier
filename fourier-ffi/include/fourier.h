#ifndef FOURIER_H_
#define FOURIER_H_

#ifdef __cplusplus

#include <complex>
#include <cstddef>
#include <memory>

#define FOURIER_COMPLEX_FLOAT_TYPE ::std::complex<float>
#define FOURIER_COMPLEX_DOUBLE_TYPE ::std::complex<double>
#define FOURIER_SIZE_TYPE ::std::size_t
#define FOURIER_STRUCT

namespace fourier {
namespace c {
extern "C" {

#else

#include <stddef.h>

#define FOURIER_COMPLEX_FLOAT_TYPE float _Complex
#define FOURIER_COMPLEX_DOUBLE_TYPE double _Complex
#define FOURIER_SIZE_TYPE size_t
#define FOURIER_STRUCT struct

#endif

enum {
  FOURIER_TRANSFORM_FFT = 0,
  FOURIER_TRANSFORM_IFFT = 1,
  FOURIER_TRANSFORM_UNSCALED_IFFT = 2,
  FOURIER_TRANSFORM_SQRT_SCALED_FFT = 3,
  FOURIER_TRANSFORM_SQRT_SCALED_IFFT = 4,
};

struct fourier_fft_float;
struct fourier_fft_double;

struct fourier_fft_float *fourier_create_float(FOURIER_SIZE_TYPE);
struct fourier_fft_double *fourier_create_double(FOURIER_SIZE_TYPE);

void fourier_destroy_float(FOURIER_STRUCT fourier_fft_float *);
void fourier_destroy_double(FOURIER_STRUCT fourier_fft_double *);

void fourier_transform_in_place_float(const FOURIER_STRUCT fourier_fft_float *,
                                      FOURIER_COMPLEX_FLOAT_TYPE *, int);
void fourier_transform_in_place_double(
    const FOURIER_STRUCT fourier_fft_double *, FOURIER_COMPLEX_DOUBLE_TYPE *,
    int);

void fourier_transform_float(const FOURIER_STRUCT fourier_fft_float *,
                             const FOURIER_COMPLEX_FLOAT_TYPE *,
                             FOURIER_COMPLEX_FLOAT_TYPE *, int);
void fourier_transform_double(const FOURIER_STRUCT fourier_fft_double *,
                              const FOURIER_COMPLEX_DOUBLE_TYPE *,
                              FOURIER_COMPLEX_DOUBLE_TYPE *, int);

#ifdef __cplusplus
} // extern "C"
} // namespace c

enum class transform {
  fft = ::fourier::c::FOURIER_TRANSFORM_FFT,
  ifft = ::fourier::c::FOURIER_TRANSFORM_IFFT,
  unscaled_ifft = ::fourier::c::FOURIER_TRANSFORM_UNSCALED_IFFT,
  sqrt_scaled_fft = ::fourier::c::FOURIER_TRANSFORM_SQRT_SCALED_FFT,
  sqrt_scaled_ifft = ::fourier::c::FOURIER_TRANSFORM_SQRT_SCALED_IFFT,
};

template <typename T> struct fft;
template <> struct fft<float> {
  explicit fft(std::size_t size)
      : impl(::fourier::c::fourier_create_float(size),
             ::fourier::c::fourier_destroy_float) {
    if (!impl)
      throw std::runtime_error("failed to initialize FFT");
  }

  fft() = delete;
  fft(const fft &) = delete;
  fft(fft &&) = default;
  fft &operator=(const fft &) = delete;
  fft &operator=(fft &&) = default;
  ~fft() = default;

  void transform_in_place(::std::complex<float> *x, transform t) const {
    ::fourier::c::fourier_transform_in_place_float(impl.get(), x,
                                                   static_cast<int>(t));
  }

  void transform(const ::std::complex<float> *in, ::std::complex<float> *out,
                 transform t) const {
    ::fourier::c::fourier_transform_float(impl.get(), in, out,
                                          static_cast<int>(t));
  }

private:
  ::std::unique_ptr<::fourier::c::fourier_fft_float,
                    void (*)(::fourier::c::fourier_fft_float *)>
      impl;
};
template <> struct fft<double> {
  explicit fft(std::size_t size)
      : impl(::fourier::c::fourier_create_double(size),
             ::fourier::c::fourier_destroy_double) {
    if (!impl)
      throw std::runtime_error("failed to initialize FFT");
  }

  fft() = delete;
  fft(const fft &) = delete;
  fft(fft &&) = default;
  fft &operator=(const fft &) = delete;
  fft &operator=(fft &&) = default;
  ~fft() = default;

  void transform_in_place(::std::complex<double> *x, transform t) const {
    ::fourier::c::fourier_transform_in_place_double(impl.get(), x,
                                                    static_cast<int>(t));
  }

  void transform(const ::std::complex<double> *in, ::std::complex<double> *out,
                 transform t) const {
    ::fourier::c::fourier_transform_double(impl.get(), in, out,
                                           static_cast<int>(t));
  }

private:
  ::std::unique_ptr<::fourier::c::fourier_fft_double,
                    void (*)(::fourier::c::fourier_fft_double *)>
      impl;
};

} // namespace fourier
#endif

#endif // FOURIER_H_
