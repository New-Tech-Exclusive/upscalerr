#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

// CUDA kernel to perform PixelShuffle (Depth-to-Space) rearrangement in FP16 NCHW layout
// Takes input of [C * S^2, H, W] and maps to [C, H * S, W * S]
__global__ void PixelShuffleKernel(
    const __half* __restrict__ input, 
    __half* __restrict__ output, 
    int inChannels, 
    int inHeight, 
    int inWidth, 
    int scaleFactor) 
{
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outC = blockIdx.z * blockDim.z + threadIdx.z;

    int outWidth = inWidth * scaleFactor;
    int outHeight = inHeight * scaleFactor;
    int outChannels = inChannels / (scaleFactor * scaleFactor);

    if (outX >= outWidth || outY >= outHeight || outC >= outChannels) return;

    // Calculate source indices
    int inX = outX / scaleFactor;
    int inY = outY / scaleFactor;
    
    int offsetX = outX % scaleFactor;
    int offsetY = outY % scaleFactor;

    // Channel index mapping: outC * S^2 + offsetY * S + offsetX
    int inC = (outC * scaleFactor * scaleFactor) + (offsetY * scaleFactor) + offsetX;

    // Spatial indexing
    int inSpatialIdx = inY * inWidth + inX;
    int inChannelOffset = inC * inHeight * inWidth;
    int inIdx = inChannelOffset + inSpatialIdx;

    int outSpatialIdx = outY * outWidth + outX;
    int outChannelOffset = outC * outHeight * outWidth;
    int outIdx = outChannelOffset + outSpatialIdx;

    output[outIdx] = input[inIdx];
}

// Host wrapper
void RunPixelShuffle(
    const void* input, 
    void* output, 
    int inChannels, 
    int inHeight, 
    int inWidth, 
    int scaleFactor, 
    cudaStream_t stream) 
{
    int outWidth = inWidth * scaleFactor;
    int outHeight = inHeight * scaleFactor;
    int outChannels = inChannels / (scaleFactor * scaleFactor);

    dim3 block(16, 16, 1);
    dim3 grid(
        (outWidth + block.x - 1) / block.x,
        (outHeight + block.y - 1) / block.y,
        (outChannels + block.z - 1) / block.z
    );

    PixelShuffleKernel<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(input),
        static_cast<__half*>(output),
        inChannels, inHeight, inWidth, scaleFactor
    );
}
