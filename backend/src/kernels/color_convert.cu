#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

// CUDA kernel to convert BGRA Surface to RGB FP16 NCHW Tensor
__global__ void ColorConvertKernel(cudaSurfaceObject_t rawSurface, __half* dstBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Read pixel from surface
    uchar4 pixel;
    surf2Dread(&pixel, rawSurface, x * 4, y);

    // Normalize directly into FP16 [0, 1]
    __half r = __float2half(static_cast<float>(pixel.z) / 255.0f); // B8G8R8A8 swizzle
    __half g = __float2half(static_cast<float>(pixel.y) / 255.0f);
    __half b = __float2half(static_cast<float>(pixel.x) / 255.0f);

    // Compute NCHW offsets
    int spatialSize = width * height;
    int idx = y * width + x;

    dstBuffer[idx] = r;                    // R channel plane
    dstBuffer[spatialSize + idx] = g;      // G channel plane
    dstBuffer[spatialSize * 2 + idx] = b;  // B channel plane
}

// CUDA kernel to copy raw input surface directly to Vulkan present surface (when SR/FG is disabled)
__global__ void CopyDirectKernel(cudaSurfaceObject_t rawSurface, cudaSurfaceObject_t presentSurface, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uchar4 pixel;
    surf2Dread(&pixel, rawSurface, x * 4, y);
    surf2Dwrite(pixel, presentSurface, x * 4, y);
}

// CUDA kernel to concatenate two RGB FP16 NCHW frames into a single 6-channel FP16 NCHW tensor
__global__ void ConcatFramesKernel(__half* framePrev, __half* frameNext, __half* dstConcat, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int spatialSize = width * height;
    int idx = y * width + x;

    // First 3 channels: framePrev
    dstConcat[idx] = framePrev[idx];
    dstConcat[spatialSize + idx] = framePrev[spatialSize + idx];
    dstConcat[spatialSize * 2 + idx] = framePrev[spatialSize * 2 + idx];

    // Next 3 channels: frameNext
    dstConcat[spatialSize * 3 + idx] = frameNext[idx];
    dstConcat[spatialSize * 4 + idx] = frameNext[spatialSize + idx];
    dstConcat[spatialSize * 5 + idx] = frameNext[spatialSize * 2 + idx];
}

// Host wrappers
void RunColorConvert(cudaSurfaceObject_t rawSurface, void* dstBuffer, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    ColorConvertKernel<<<grid, block, 0, stream>>>(rawSurface, static_cast<__half*>(dstBuffer), width, height);
}

void RunCopyDirect(cudaSurfaceObject_t rawSurface, cudaSurfaceObject_t presentSurface, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    CopyDirectKernel<<<grid, block, 0, stream>>>(rawSurface, presentSurface, width, height);
}

void RunConcatFrames(void* framePrev, void* frameNext, void* dstConcat, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    ConcatFramesKernel<<<grid, block, 0, stream>>>(static_cast<__half*>(framePrev), static_cast<__half*>(frameNext), static_cast<__half*>(dstConcat), width, height);
}
