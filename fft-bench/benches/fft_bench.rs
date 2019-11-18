use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num::Complex;
use rand::distributions::Standard;
use rand::Rng;

fn pow2_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT, f32, powers of 2");
    for size in (1..10).map(|x| 2usize.pow(x)) {
        let input = rand::thread_rng()
            .sample_iter(&Standard)
            .zip(rand::thread_rng().sample_iter(&Standard))
            .take(size)
            .map(|(x, y)| Complex::new(x, y))
            .collect::<Vec<_>>();

        // My FFT
        let mut my_fft = fft::Fft32::new(size);
        group.bench_with_input(BenchmarkId::new("Mine", size), &input, |b, i| {
            let mut input = Vec::new();
            input.extend_from_slice(i);
            b.iter(|| my_fft.fft_in_place(input.as_mut()))
        });

        // RustFFT
        let rustfft = rustfft::FFTplanner::<f32>::new(false).plan_fft(size);
        group.bench_with_input(BenchmarkId::new("RustFFT", size), &input, |b, i| {
            let mut input = Vec::new();
            input.extend_from_slice(i);
            let mut output = vec![Complex::default(); input.len()];
            b.iter(|| rustfft.process(input.as_mut(), output.as_mut()))
        });
    }
    group.finish();
}

criterion_group!(benches, pow2_f32);
criterion_main!(benches);
