import operator
import unittest
import multiprocessing
import timeit
from ctypes import c_byte, c_char, c_ubyte
from functools import reduce
from threading import Thread
from typing import Tuple

from multiprocessing.sharedctypes import RawArray

import matplotlib.pyplot as plt

import cv2
import numpy as np
import random
from ops import *

# Number of runs to average benchmark data
BENCHMARK_COUNT = 5


def get_images(count):
    with open("dataset/files.txt") as files:
        filenames = files.readlines()
        images = []
        for i in range(count):
            image_filename = random.choice(filenames)
            image = cv2.imread("dataset/" + image_filename.strip())
            images.append(image)

        return images


def proc_image(img):
    return benchmark_op(img)


def to_raw_array(image: np.ndarray):
    data_type = image.dtype
    shape = image.shape
    size = 1
    for dim in shape:
        size *= dim
    raw_array = RawArray(c_ubyte, size)

    raw_array_np = np.frombuffer(raw_array, dtype=np.uint8).reshape(shape)
    np.copyto(raw_array_np, image)

    return raw_array, shape


def from_raw_array(raw_array: RawArray, shape: Tuple):
    image = np.frombuffer(raw_array, dtype=np.uint8).reshape(shape)
    return image


def proc_from_raw_array(name):
    arr_and_shape = raw_arrays[name]
    raw_arr, shape = arr_and_shape[0], arr_and_shape[1]

    img = from_raw_array(raw_arr, shape)

    return benchmark_op(img)


def put_raw_array(raw_array, shape):
    name = str(len(raw_arrays) + 1)
    raw_arrays[name] = (raw_array, shape)
    return name


raw_arrays = {}


def test_process_sequential(num_images):
    images = get_images(num_images)

    def blur_all():
        blurred = [proc_image(img) for img in images]

    op_time = timeit.timeit(blur_all, number=BENCHMARK_COUNT) / BENCHMARK_COUNT

    print("Avg time to process {} images (sequential): {} sec".format(num_images, op_time))
    return op_time


def test_process_parallel(num_images, workers=12):
    images = get_images(num_images)

    raw_images = [to_raw_array(img) for img in images]
    arr_names = [put_raw_array(*raw_arr) for raw_arr in raw_images]

    with multiprocessing.Pool(workers) as p:
        def proc_all():
            p.map(proc_from_raw_array, arr_names)
        # spool up or something? For some reason this makes it faster:
        proc_all()
        op_time = timeit.timeit(proc_all, number=BENCHMARK_COUNT) / BENCHMARK_COUNT

    print("Avg time to process {} images (parallel, {} worker threads): {} sec".format(num_images, workers, op_time))
    return op_time


def test_process_sequential_shared(num_images):
    images = get_images(num_images)

    raw_images = [to_raw_array(img) for img in images]
    arr_names = [put_raw_array(*raw_arr) for raw_arr in raw_images]

    def proc_all():
        [proc_from_raw_array(name) for name in arr_names]

    op_time = timeit.timeit(proc_all, number=BENCHMARK_COUNT) / BENCHMARK_COUNT

    print("Avg time to process {} images (sequential - shared mem): {} sec".format(num_images, op_time))
    return op_time


def test_process_threaded(num_images):
    images = get_images(num_images)

    # This sucks because it has to re-create the threads every time, yet it's still the fastest?!
    def proc_all():
        img_threads = []
        for image in images:
            img_threads.append(Thread(target=proc_image, args=(image,), daemon=True))

        for thread in img_threads:
            thread.start()

        for thread in img_threads:
            thread.join()

    op_time = timeit.timeit(proc_all, number=BENCHMARK_COUNT) / BENCHMARK_COUNT

    print("Avg time to process {} images (sequential - threading/no multiprocessing): {} sec".format(num_images, op_time))
    return op_time


if __name__ == '__main__':
    data = {'multiprocessing': [], 'sequential': [], 'sequential-shared memory': [], 'threading': []}

    for i in range(1, 50, 2):
        data['multiprocessing'].append((i, test_process_parallel(i)))
        data['sequential'].append((i, test_process_sequential(i)))
        data['sequential-shared memory'].append((i, test_process_sequential_shared(i)))
        data['threading'].append((i, test_process_threaded(i)))

    for name, values in data.items():
        plt.plot([val[0] for val in values], [val[1] for val in values], label=name)
    plt.xlabel("Number of images")
    plt.ylabel("Processing time")
    plt.legend()
    plt.savefig("benchmark.png")
