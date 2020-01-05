#ifndef FOURIER_H_
#define FOURIER_H_

#ifdef __cplusplus

#include <complex>
#include <cstddef>
#include <memory>

#define FOURIER_COMPLEX_FLOAT_TYPE ::std::complex<float>
#define FOURIER_COMPLEX_DOUBLE_TYPE ::std::complex<double>
#define FOURIER_SIZE_TYPE ::std::size_t
#define FOURIER_BOOL_TYPE bool
#define FOURIER_STRUCT

namespace fourier {
namespace detail {
extern "C" {

#else

#include <stddef.h>

#define FOURIER_COMPLEX_FLOAT_TYPE float _Complex
#define FOURIER_COMPLEX_DOUBLE_TYPE double _Complex
#define FOURIER_SIZE_TYPE size_t
#define FOURIER_BOOL_TYPE _Bool
#define FOURIER_STRUCT struct

#endif

struct fourier_fft_float;
struct fourier_fft_double;

struct fourier_fft_float *fourier_create_float(FOURIER_SIZE_TYPE);
struct fourier_fft_double *fourier_create_double(FOURIER_SIZE_TYPE);

void fourier_destroy_float(FOURIER_STRUCT fourier_fft_float *);
void fourier_destroy_double(FOURIER_STRUCT fourier_fft_double *);

void fourier_transform_in_place_float(const FOURIER_STRUCT fourier_fft_float *,
                                      FOURIER_COMPLEX_FLOAT_TYPE *,
                                      FOURIER_BOOL_TYPE);
void fourier_transform_in_place_double(
    const FOURIER_STRUCT fourier_fft_double *, FOURIER_COMPLEX_DOUBLE_TYPE *,
    FOURIER_BOOL_TYPE);

void fourier_transform_float(const FOURIER_STRUCT fourier_fft_float *,
                             const FOURIER_COMPLEX_FLOAT_TYPE *,
                             FOURIER_COMPLEX_FLOAT_TYPE *, FOURIER_BOOL_TYPE);
void fourier_transform_double(const FOURIER_STRUCT fourier_fft_double *,
                              const FOURIER_COMPLEX_DOUBLE_TYPE *,
                              FOURIER_COMPLEX_DOUBLE_TYPE *, FOURIER_BOOL_TYPE);

#ifdef __cplusplus
} // extern "C"
} // namespace detail

template <typename T> struct fft;
template <> struct fft<float> {
  explicit fft(std::size_t size)
      : impl(::fourier::detail::fourier_create_float(size),
             ::fourier::detail::fourier_destroy_float) {}

  fft() = delete;
  fft(const fft &) = delete;
  fft(fft &&) = default;
  fft &operator=(const fft &) = delete;
  fft &operator=(fft &&) = default;
  ~fft() = default;

  void transform_in_place(::std::complex<float> *x, bool forward) {
    ::fourier::detail::fourier_transform_in_place_float(impl.get(), x, forward);
  }

  void transform(const ::std::complex<float> *in, ::std::complex<float> *out,
                 bool forward) const {
    ::fourier::detail::fourier_transform_float(impl.get(), in, out, forward);
  }

private:
  ::std::unique_ptr<::fourier::detail::fourier_fft_float,
                    void (*)(::fourier::detail::fourier_fft_float *)>
      impl;
};
template <> struct fft<double> {
  explicit fft(std::size_t size)
      : impl(::fourier::detail::fourier_create_double(size),
             ::fourier::detail::fourier_destroy_double) {}

  fft() = delete;
  fft(const fft &) = delete;
  fft(fft &&) = default;
  fft &operator=(const fft &) = delete;
  fft &operator=(fft &&) = default;
  ~fft() = default;

  void transform_in_place(::std::complex<double> *x, bool forward) {
    ::fourier::detail::fourier_transform_in_place_double(impl.get(), x,
                                                         forward);
  }

  void transform(const ::std::complex<double> *in, ::std::complex<double> *out,
                 bool forward) const {
    ::fourier::detail::fourier_transform_double(impl.get(), in, out, forward);
  }

private:
  ::std::unique_ptr<::fourier::detail::fourier_fft_double,
                    void (*)(::fourier::detail::fourier_fft_double *)>
      impl;
};

} // namespace fourier
#endif

#endif // FOURIER_H_
