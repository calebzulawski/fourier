use libc::{c_int, size_t};

fn convert_transform(code: c_int) -> fourier::Transform {
    match code {
        0 => fourier::Transform::Fft,
        1 => fourier::Transform::Ifft,
        2 => fourier::Transform::UnscaledIfft,
        3 => fourier::Transform::SqrtScaledFft,
        4 => fourier::Transform::SqrtScaledIfft,
        _ => panic!("unknown transform code"),
    }
}

fn create<T: fourier::Float>(size: usize) -> *const Box<dyn fourier::Fft<Real = T> + Send> {
    std::panic::catch_unwind(|| Box::into_raw(Box::new(fourier::create_fft(size))))
        .unwrap_or(std::ptr::null_mut())
}

unsafe fn destroy<T: fourier::Float>(state: *mut Box<dyn fourier::Fft<Real = T> + Send>) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Box::from_raw(state);
    }));
}

unsafe fn transform_in_place<T: fourier::Float>(
    state: *const Box<dyn fourier::Fft<Real = T> + Send>,
    input: *mut num_complex::Complex<T>,
    transform: c_int,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (*state).transform_in_place(
            std::slice::from_raw_parts_mut(input, (*state).size()),
            convert_transform(transform),
        );
    }));
}

unsafe fn transform<T: fourier::Float>(
    state: *const Box<dyn fourier::Fft<Real = T> + Send>,
    input: *const num_complex::Complex<T>,
    output: *mut num_complex::Complex<T>,
    transform: c_int,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (*state).transform(
            std::slice::from_raw_parts(input, (*state).size()),
            std::slice::from_raw_parts_mut(output, (*state).size()),
            convert_transform(transform),
        );
    }));
}

/// Create an FFT over f32.
#[no_mangle]
pub extern "C" fn fourier_create_float(
    size: usize,
) -> *const Box<dyn fourier::Fft<Real = f32> + Send> {
    create(size)
}

/// Destroy an FFT over f32.
///
/// # Safety
/// `state` must have been created by `fourier_create_float`.
#[no_mangle]
pub unsafe extern "C" fn fourier_destroy_float(
    state: *mut Box<dyn fourier::Fft<Real = f32> + Send>,
) {
    destroy(state)
}

/// Perform an in-place FFT over f32.
///
/// # Safety
/// `state` must have been created by `fourier_create_float`.
/// `input` must point to an array with length equal to the FFT length.
#[no_mangle]
pub unsafe extern "C" fn fourier_transform_in_place_float(
    state: *const Box<dyn fourier::Fft<Real = f32> + Send>,
    input: *mut num_complex::Complex<f32>,
    transform: c_int,
) {
    transform_in_place(state, input, transform)
}

/// Perform an in-place FFT over f32.
///
/// # Safety
/// `state` must have been created by `fourier_create_float`.
/// `input` must point to an array with length equal to the FFT length.
#[no_mangle]
pub unsafe extern "C" fn fourier_transform_float(
    state: *const Box<dyn fourier::Fft<Real = f32> + Send>,
    input: *const num_complex::Complex<f32>,
    output: *mut num_complex::Complex<f32>,
    transform: c_int,
) {
    self::transform(state, input, output, transform)
}

/// Create an FFT over f64.
#[no_mangle]
pub extern "C" fn fourier_create_double(
    size: size_t,
) -> *const Box<dyn fourier::Fft<Real = f64> + Send> {
    create(size)
}

/// Destroy an FFT over f64.
///
/// # Safety
/// `state` must have been created by `fourier_create_double`.
#[no_mangle]
pub unsafe extern "C" fn fourier_destroy_double(
    state: *mut Box<dyn fourier::Fft<Real = f64> + Send>,
) {
    destroy(state)
}

/// Perform an in-place FFT over f64.
///
/// # Safety
/// `state` must have been created by `fourier_create_double`.
/// `input` must point to an array with length equal to the FFT length.
#[no_mangle]
pub unsafe extern "C" fn fourier_transform_in_place_double(
    state: *const Box<dyn fourier::Fft<Real = f64> + Send>,
    input: *mut num_complex::Complex<f64>,
    transform: c_int,
) {
    transform_in_place(state, input, transform)
}

/// Perform an out-of-place FFT over f64.
///
/// # Safety
/// `state` must have been created by `fourier_create_double`.
/// `input` and `output` must point to arrays with length equal to the FFT length.
#[no_mangle]
pub unsafe extern "C" fn fourier_transform_double(
    state: *const Box<dyn fourier::Fft<Real = f64> + Send>,
    input: *const num_complex::Complex<f64>,
    output: *mut num_complex::Complex<f64>,
    transform: c_int,
) {
    self::transform(state, input, output, transform)
}
