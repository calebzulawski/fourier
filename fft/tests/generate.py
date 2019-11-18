#!/usr/bin/env python3
import numpy
import scipy.linalg
import json

def jsonify_cplx(z):
    return (z.real, z.imag)

numpy.random.seed(1234)

sizes = [
    2, 4, 8, 16, 32, 64, 128, 256,   # powers of 2
    3, 9, 27, 81, 243,               # powers of 3
    6, 12, 18, 36, 54, 72, 108, 216  # mixed powers of 2 and 3
]

for size in sizes:
    x = numpy.random.normal(size=size) + numpy.random.normal(size=size)*1j
    #y = numpy.fft.fft(x)
    y = scipy.linalg.dft(x)
    data = {}
    data["x"] = list(map(jsonify_cplx, x))
    data["y"] = list(map(jsonify_cplx, y))
    with open("vectors/{}.json".format(size), "w") as jsonfile:
        json.dump(data, jsonfile)

with open("vectors/generate_tests.rs", "w") as testfile:
    for size in sizes:
        testfile.write("generate_vector_test!{{@forward_f32 forward_f32_{}, \"{}.json\"}}\n".format(size, size))
        testfile.write("generate_vector_test!{{@inverse_f32 inverse_f32_{}, \"{}.json\"}}\n".format(size, size))
