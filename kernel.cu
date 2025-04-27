
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cassert>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



//for calculating global index 
// 


__global__ void addVectorsGPU(int* d_a, int* d_b, int* d_c, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        d_c[tid] = d_a[tid] + d_b[tid];

    }
}


void addTwoArrays(int *h_a, int *h_b, int *h_c, int size)
{
    int blocksize = 256;


    int numblocks = (size + blocksize - 1) / blocksize;

    //device vectors

    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    //copy memory from host to the device

    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    addVectorsGPU << <numblocks, blocksize , blocksize * sizeof(int) >> > (d_a, d_b, d_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; i++)
    {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;

    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}


void addTwoVectors(vector<int>& h_a , vector<int>& h_b , vector<int>& h_c, int size)
 {
    int blocksize = 256;
    

    int numblocks = (size + blocksize - 1) / blocksize;

    //device vectors

    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    //copy memory from host to the device

    cudaMemcpy(d_a, h_a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    addVectorsGPU << <numblocks, blocksize >> > (d_a, d_b, d_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; i++)
    {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;

    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}


void addTwoVectorsUnifedMemory(vector<int>& h_a, vector<int>& h_b, vector<int>& h_c, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    size_t bytes = size * sizeof(int);

    // Allocate Unified Memory
    int* d_a, * d_b, * d_c;
    cudaMallocManaged(&d_a, bytes);
    cudaMallocManaged(&d_b, bytes);
    cudaMallocManaged(&d_c, bytes);

    // Copy data from host vectors to Unified Memory
    for (int i = 0; i < size; i++) {
        d_a[i] = h_a[i];
        d_b[i] = h_b[i];
    }

    // Launch kernel
    addVectorsGPU << <numBlocks, blockSize >> > (d_a, d_b, d_c, size);

    // Synchronize to wait for GPU computation to finish
    cudaDeviceSynchronize();

    // Copy result back to host vector
    h_c.resize(size);
    for (int i = 0; i < size; i++) {
        h_c[i] = d_c[i];
        cout << h_c[i] << endl; 
    }

    // Free Unified Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void matrixAddElement(float* a, float* b, float* c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.y + threadIdx.x;
}

__global__ void matrixAddRow(float* A, float* B, float* C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int col = 0; col < n; col++) {
            A[row * n + col] = B[row * n + col] + C[row * n + col];
        }
    }
}

// Kernel: Each thread produces one output column
__global__ void matrixAddColumn(float* A, float* B, float* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        for (int row = 0; row < n; row++) {
            A[row * n + col] = B[row * n + col] + C[row * n + col];
        }
    }
}

__global__ void matrixMultiGlobalMemory(int* A, int* B, int* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
   
    //boundry check
    if (row < n && col < n)
    {
        int tmp = 0;
        for (int i = 0; i < n; i++)
        {
            tmp += A[row * n + i] * B[i * n + col];

        }
        C[row * n + col] = tmp;
    }
}

__global__ void matrixMultiSharedMemory(int* A, int* B, int* C, int n)
{
    //define shared memory
    __shared__ int Ashared[16 * 16];
    __shared__ int Bshared[16 * 16];

    //define global index for row & col
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    //allocate the shared memory from the original arrays
    int tmp = 0;
    for (int i = 0; i < (n + dim - 1) / dim; i++)
    {
        Ashared[ty * dim + tx] = A[(row * n) + (i * dim) + tx];
        Bshared[ty * dim + tx] = B[(i * dim * n) + (ty * n) + col];
        
        //__synchthreads();

        for (size_t j = 0; j < dim; j++)
        {
            tmp += Ashared[ty * dim + j] * Bshared[j * dim + tx];
        }
    }
}

void verifyMatrixMulti(int* A, int* B, int* C, int n)
{
    int tmp;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmp = 0;
            for (int k = 0; k < n; k++)
            {
                tmp += A[i * n + k] * B[k * n + j];
            }
            assert(tmp == C[i * n + j]);
        }

    }
}

void init_matrix(int* a, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        a[i] = rand() % 100;
    }
}


void matrixAddHost(float* A, float* B, float* C, int n, int kernelChoice) {
    int size = n * n * sizeof(float); // Total size of a matrix

    // Device pointers
    float* d_A, * d_B, * d_C;

    // Allocate memory on the device for input and output matrices
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // Execution configuration parameters (adjust based on kernel design)
    dim3 blockDim(16, 16);  // 16x16 block size
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the chosen kernel
    switch (kernelChoice) {
    case 1:
        // Kernel where each thread computes one element
        matrixAddElement << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
        break;
    case 2:
        // Kernel where each thread computes one row
        blockDim = dim3(16);
        gridDim = dim3((n + blockDim.x - 1) / blockDim.x);
        matrixAddRow << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
        break;
    case 3:
        // Kernel where each thread computes one column
        blockDim = dim3(16);
        gridDim = dim3((n + blockDim.x - 1) / blockDim.x);
        matrixAddColumn << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
        break;
    default:
        std::cerr << "Invalid kernel choice" << std::endl;
        return;
    }

    // Copy output data from device to host
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Helper function to initialize matrices
void initializeMatrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Helper function to print matrices
void printMatrix(float* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

void OpenCVCudaBlurredImg()
{
    cv::Mat img = cv::imread("D:/dragon.jpg", cv::IMREAD_GRAYSCALE);

    // now we will upload the image to the gpu

    cv::cuda::GpuMat d_img;

    d_img.upload(img);

    // Create a Gaussian filter using CUDA
    cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(d_img.type(), d_img.type(), cv::Size(15, 15), 3);

    // Apply the Gaussian filter
    cv::cuda::GpuMat d_blurred;
    gaussianFilter->apply(d_img, d_blurred);

    // Download the blurred image back to CPU
    cv::Mat blurred;
    d_blurred.download(blurred);

    // Display the result
    cv::imshow("Blurred Image", blurred);
    cv::waitKey(0);

}

void OpenCVcudaBlurredImg2()
{

    cv::Mat h_img = cv::imread("path", cv::IMREAD_GRAYSCALE);

    cv::cuda::GpuMat d_img;

    d_img.upload(h_img);

    //create the filter

    cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(d_img.type(), d_img.type(), cv::Size(15, 15), 3);

    //apply the filter
    cv::cuda::GpuMat d_blurred_img;
    gaussianFilter->apply(d_img, d_blurred_img);
    cv::Mat blurred;
    d_blurred_img.download(blurred);

    // Display the result
    cv::imshow("Blurred Image", blurred);
    cv::waitKey(0);
}
int main()
{
    int size = 16;
    vector<int> h_a(size, 1); // Initialize h_a with 1s
    vector<int> h_b(size, 2); // Initialize h_b with 2s
    vector<int> h_c;          // Result vector

    addTwoVectorsUnifedMemory(h_a, h_b, h_c, size);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


