# CUDA Vector and Matrix Operations

## Overview

This project implements several CUDA-accelerated functions for basic vector and matrix operations using both global memory and shared memory approaches.  
Additionally, it demonstrates simple GPU-accelerated image processing using **OpenCV's CUDA module**.

The main focus is on:
- Adding two vectors using CUDA kernels.
- Performing matrix addition and multiplication with different parallelization strategies.
- Demonstrating unified memory usage for easier memory management between CPU and GPU.
- Applying a **Gaussian Blur** filter to an image using OpenCV's CUDA API.

## Features

- **Vector Addition**
  - Traditional CUDA memory management.
  - Unified Memory usage for simplified data handling.
  
- **Matrix Addition**
  - Element-wise parallelization.
  - Row-wise and Column-wise parallelization.

- **Matrix Multiplication**
  - Using global memory access.
  - Optimized version using shared memory for better performance.

- **OpenCV CUDA Example**
  - GPU-accelerated image blurring using Gaussian filters with OpenCV.

## Project Structure

| File/Function                     | Description                                                        |
|------------------------------------|--------------------------------------------------------------------|
| `addVectorsGPU`                    | Kernel to add two integer arrays element-wise.                    |
| `addTwoArrays`                     | Host function to add two arrays using CUDA (manual memory copy).   |
| `addTwoVectorsUnifiedMemory`       | Host function to add two vectors using CUDA unified memory.        |
| `matrixAddElement`, `matrixAddRow`, `matrixAddColumn` | Different matrix addition kernels based on parallelization strategy. |
| `matrixMultiGlobalMemory`          | Matrix multiplication using global memory.                        |
| `matrixMultiSharedMemory`          | Matrix multiplication using shared memory optimization.           |
| `OpenCVCudaBlurredImg`             | Demonstrates blurring an image on the GPU using OpenCV.            |

## Prerequisites

- **CUDA Toolkit** installed (tested with CUDA 11+).
- **OpenCV** compiled with **CUDA support**.
- **C++ Compiler** supporting C++11 or newer (e.g., Visual Studio, GCC).
- A CUDA-capable GPU.

## How to Build and Run

1. **Clone the Repository**
   ```bash
   git clone <your-repo-link>
   cd <project-folder>
