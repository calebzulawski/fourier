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

#[no_mangle]
pub extern "C" fn fourier_create_float(size: usize) -> *mut (dyn fourier::Fft<Real = f32> + Send) {
    Box::into_raw(Box::new(fourier::create_fft_f32(size)))
}

#[no_mangle]
pub unsafe extern "C" fn fourier_destroy_float(state: *mut (dyn fourier::Fft<Real = f32> + Send)) {
    Box::from_raw(state);
}

#[no_mangle]
pub unsafe extern "C" fn fourier_transform_in_place_float(
    state: *const (dyn fourier::Fft<Real = f32> + Send),
    input: *mut num_complex::Complex<f32>,
    transform: c_int,
) {
    (*state).transform_in_place(
        std::slice::from_raw_parts_mut(input, (*state).size()),
        convert_transform(transform),
    );
}

#[no_mangle]
pub unsafe extern "C" fn fourier_transform_float(
    state: *const Box<dyn fourier::Fft<Real = f32> + Send>,
    input: *const num_complex::Complex<f32>,
    output: *mut num_complex::Complex<f32>,
    transform: c_int,
) {
    (*state).transform(
        std::slice::from_raw_parts(input, (*state).size()),
        std::slice::from_raw_parts_mut(output, (*state).size()),
        convert_transform(transform),
    );
}

#[no_mangle]
pub extern "C" fn fourier_create_double(
    size: size_t,
) -> *mut (dyn fourier::Fft<Real = f64> + Send) {
    Box::into_raw(Box::new(fourier::create_fft_f64(size)))
}

#[no_mangle]
pub unsafe extern "C" fn fourier_destroy_double(state: *mut (dyn fourier::Fft<Real = f64> + Send)) {
    Box::from_raw(state);
}

#[no_mangle]
pub unsafe extern "C" fn fourier_transform_in_place_double(
    state: *const Box<dyn fourier::Fft<Real = f64> + Send>,
    input: *mut num_complex::Complex<f64>,
    transform: c_int,
) {
    (*state).transform_in_place(
        std::slice::from_raw_parts_mut(input, (*state).size()),
        convert_transform(transform),
    );
}

#[no_mangle]
pub unsafe extern "C" fn fourier_transform_double(
    state: *const Box<dyn fourier::Fft<Real = f64> + Send>,
    input: *const num_complex::Complex<f64>,
    output: *mut num_complex::Complex<f64>,
    transform: c_int,
) {
    (*state).transform(
        std::slice::from_raw_parts(input, (*state).size()),
        std::slice::from_raw_parts_mut(output, (*state).size()),
        convert_transform(transform),
    );
}
