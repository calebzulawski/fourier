use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num::Complex;
use rand::distributions::Standard;
use rand::Rng;

macro_rules! create_bench {
    {
        $name:ident, $type:ty, $fourier:path, $fftw:path, $fftw_type:path
    } => {
        fn $name(
            c: &mut Criterion,
            title: &str,
            sizes: &mut dyn std::iter::Iterator<Item = usize>,
            forward: bool,
        ) {
            let mut group = c.benchmark_group(title);
            for size in sizes {
                let input = rand::thread_rng()
                    .sample_iter(&Standard)
                    .zip(rand::thread_rng().sample_iter(&Standard))
                    .take(size)
                    .map(|(x, y)| Complex::new(x, y))
                    .collect::<Vec<_>>();

                // Fourier
                let fourier = $fourier(size);
                group.bench_with_input(BenchmarkId::new("Fourier", size), &input, |b, i| {
                    let mut input = Vec::new();
                    input.extend_from_slice(i);
                    let mut output = vec![Complex::default(); input.len()];
                    let transform = if forward {
                        fourier::Transform::Fft
                    } else {
                        fourier::Transform::Ifft
                    };
                    b.iter(|| fourier.transform(&input, &mut output, transform));
                });

                // RustFFT
                let rustfft = rustfft::FFTplanner::<$type>::new(!forward).plan_fft(size);
                group.bench_with_input(BenchmarkId::new("RustFFT", size), &input, |b, i| {
                    let mut input = vec![Default::default(); i.len()];
                    for (i, o) in i.iter().zip(input.iter_mut()) {
                        *o = num_02::Complex::new(i.re, i.im);
                    }
                    let mut output = vec![num_02::Complex::default(); input.len()];
                    b.iter(|| rustfft.process(input.as_mut(), output.as_mut()))
                });

                // FFTW
                use fftw::plan::C2CPlan;
                let mut fftw = $fftw(
                    &[size],
                    if forward {
                        fftw::types::Sign::Forward
                    } else {
                        fftw::types::Sign::Backward
                    },
                    fftw::types::Flag::MEASURE,
                )
                .unwrap();
                group.bench_with_input(BenchmarkId::new("FFTW", size), &input, |b, i| {
                    let mut input = fftw::array::AlignedVec::new(size);
                    for (i, o) in i.iter().zip(input.as_slice_mut().iter_mut()) {
                        *o = $fftw_type(i.re, i.im);
                    }
                    let mut output = fftw::array::AlignedVec::new(size);
                    b.iter(|| {
                        fftw.c2c(input.as_slice_mut(), output.as_slice_mut())
                            .unwrap()
                    })
                });
            }
            group.finish();
        }
    }
}

macro_rules! create_scenarios {
    {
        $([$name:ident, $title:literal, $sizes:expr])*
    } => {
        mod bench_f32 {
            use super::*;
            create_bench! { bench, f32, fourier::create_fft::<f32>, fftw::plan::C2CPlan32::aligned, fftw::types::c32::new }
            pub mod fft {
                use super::*;
                $(
                pub fn $name(c: &mut Criterion) {
                    bench(
                        c,
                        &format!("FFT, f32, {}", $title),
                        $sizes,
                        true,
                    )
                }
                )*
            }
            pub mod ifft {
                use super::*;
                $(
                pub fn $name(c: &mut Criterion) {
                    bench(
                        c,
                        &format!("IFFT, f32, {}", $title),
                        $sizes,
                        false,
                    )
                }
                )*
            }
        }
        mod bench_f64 {
            use super::*;
            create_bench! { bench, f64, fourier::create_fft::<f64>, fftw::plan::C2CPlan64::aligned, fftw::types::c64::new }
            pub mod fft {
                use super::*;
                $(
                pub fn $name(c: &mut Criterion) {
                    bench(
                        c,
                        &format!("FFT, f64, {}", $title),
                        $sizes,
                        true,
                    )
                }
                )*
            }
            pub mod ifft {
                use super::*;
                $(
                pub fn $name(c: &mut Criterion) {
                    bench(
                        c,
                        &format!("IFFT, f64, {}", $title),
                        $sizes,
                        false,
                    )
                }
                )*
            }
        }
        criterion_group!(
            benches,
            $(
            bench_f32::fft::$name,
            bench_f32::ifft::$name,
            bench_f64::fft::$name,
            bench_f64::ifft::$name,
            )*
        );
    }
}

create_scenarios! {
    [pow2, "powers of two", &mut (8..11).map(|x| 2usize.pow(x))]
    [pow3, "powers of three", &mut (5..8).map(|x| 3usize.pow(x))]
    [pow5, "powers of five", &mut (3..6).map(|x| 5usize.pow(x))]
    [composite, "composites of 2, 3, 5", &mut [222, 722, 1418].iter().map(|x| *x)]
    [prime, "primes", &mut [191, 439, 1013].iter().map(|x| *x)]
}

criterion_main!(benches);
