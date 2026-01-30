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

import cv2

# Data Structures define - class #


# Functions define #


# Closure & Decorator


# Main function define #

def main(*args, **kwargs):
    lib = ctypes.cdll.LoadLibrary("lib/libcuda2py.so")

    lib.cuda_convolution.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    img = cv2.imread("img/original.png", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int32)

    n = 2048; m = 3; k = n + m - 1

    kernels = {
        "edge_detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int32),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int32),
    }

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y1 = cy - n // 2; y2 = y1 + n
    x1 = cx - n // 2; x2 = x1 + n
    gray = gray[y1:y2, x1:x2]

    for name, kernel in kernels.items():
        c = np.zeros((k, k), dtype=np.int32)
        lib.cuda_convolution(gray.ravel(), kernel.ravel(), c.ravel(),
                             n, m, k)
        c = cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("img/" + name + ".png", c)

# EP
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
