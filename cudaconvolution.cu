// This file includes GPU convolution function
// and CPU image filtering function that uses GPU convolution.
//
// XXX: There is no error handling.
//

#include <stdint.h>
#include <stdio.h>

const int DEVICE_ID = 0;
const int MAX_DEVICE_BLOCK_PER_DIM_CNT  = 65535;

// This function converts 2d coordinates to 1d coordinates
//
extern "C" __device__ int32_t to1d(int32_t x, int32_t y, int32_t w) {
    return x * w + y;
}

// This function performs convolution on GPU
//
extern "C" __global__ void convolve(const uint32_t *src, uint32_t *dst,
		                    const int32_t data_h, const int32_t data_w,
                                    const float *kernel,
                                    const int k_h, const int k_w) {
    int global_x = blockIdx.x;
    int global_y = blockIdx.y;

    int k_len = k_h * k_w;

    extern __shared__ float cache[];

    while (global_x < data_h && global_y < data_w) {
	    // Reset shared memory
	    for (int i = 0; i < k_len; ++i) {
	        cache[i] = 0;
	    }

	    // Convolution
	    int right_up_kernel_corner_x = global_x - k_h / 2;
	    int right_up_kernel_corner_y = global_y - k_w / 2;

	    int cur_x = right_up_kernel_corner_x + threadIdx.x;
	    int cur_y = right_up_kernel_corner_y + threadIdx.y;

            int img_idx = to1d(cur_x, cur_y, data_w);
            int k_idx = to1d(threadIdx.x, threadIdx.y, k_w);

	    if (cur_x >= 0 && cur_x < data_h && cur_y >= 0 && cur_y < data_w) {
		cache[k_idx] = (float)src[img_idx] * kernel[k_idx];
	    }
        __syncthreads();

	    // Reduction
	    int cur_k_len = k_len;
	    while (cur_k_len > 1) {
	        if (k_idx <= (cur_k_len + 1) / 2) {
                int k_from_idx = k_idx + (cur_k_len + 1) / 2;
                if (k_idx < k_len) {
                   cache[k_idx] += cache[k_from_idx];
                }
            }
            cur_k_len = (cur_k_len + 1) / 2;
            __syncthreads();
	    }
	    __syncthreads();

	    // Write
	    dst[to1d(global_x, global_y, data_w)] = (int)cache[0];
        __syncthreads();

        // Next block
        global_x += blockDim.x;
        global_y += blockDim.y;
    }
}

// This function performes one-channel image filtering
//
extern "C" int filter(const uint32_t *src, uint32_t *dst,
		              const int32_t data_h, const int32_t data_w,
		              const float *kernel,
		              const int32_t k_h, const int32_t k_w) {
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(DEVICE_ID);
    // printf("CUDA Device Count: %d\n", gpu_count);
    // printf("Using Device %d\n\n", DEVICE_ID);

    // Copy arrays
    int img_len = data_h * data_w;
    int k_len = k_h * k_w;

    uint32_t *dev_src;
    cudaMalloc(&dev_src, sizeof(uint32_t) * img_len);
    cudaMemcpy(dev_src, src, sizeof(uint32_t) * img_len, cudaMemcpyHostToDevice);

    uint32_t *dev_dst;
    cudaMalloc(&dev_dst, sizeof(uint32_t) * img_len);

    float *dev_kernel;
    cudaMalloc(&dev_kernel, sizeof(float) * k_len);
    cudaMemcpy(dev_kernel, kernel, sizeof(float) * k_len, cudaMemcpyHostToDevice);

    // Convolve
    dim3 gridSize = dim3(min(data_h, MAX_DEVICE_BLOCK_PER_DIM_CNT),
		                 min(data_w, MAX_DEVICE_BLOCK_PER_DIM_CNT), 1);
    dim3 blockSize = dim3(k_h, k_w, 1);
    convolve<<<gridSize, blockSize, sizeof(float) * k_len>>>(dev_src, dev_dst, data_h, data_w,
                                                             dev_kernel, k_h, k_w);
    cudaDeviceSynchronize();

    // Get results
    cudaMemcpy(dst, dev_dst, sizeof(uint32_t) * img_len, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(dev_kernel);

    return 0;
}
