#!/usr/bin/env python3
""" main.py
Description

Date
Jan 27, 2026
"""
__author__ = "Jeong Hoon (Sian) Choi"
__copyright__ = "Copyright 2024 Jeong Hoon Choi"
__license__ = "MIT"
__version__ = "1.0.0"

# Import #
import os, sys
import ctypes
import numpy as np
import time

# Data Structures define - class #


# Functions define #


# Closure & Decorator


# Main function define #

def main(*args, **kwargs):
    lib = ctypes.cdll.LoadLibrary("lib/libcuda2py.so")

    lib.cuda_matrix_multiplication.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    n = 1024; m = 1024; k = 1024
    a = np.random.rand(n, k).astype(np.float32)
    b = np.random.rand(k, m).astype(np.float32)
    c = np.zeros((n, m), dtype=np.float32)

    start = time.time()
    lib.cuda_matrix_multiplication(a.ravel(), b.ravel(), c.ravel(), n, m, k)
    end = time.time()

    print(f"Python call to CUDA library cost time: {end-start:.4f}")

# EP
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
