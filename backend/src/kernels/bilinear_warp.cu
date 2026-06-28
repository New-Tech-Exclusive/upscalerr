#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

// CUDA kernel performing dense bilinear warping using optical flow vectors
// Warps srcFrame by flowField (scaled by timestep=0.5) to produce dstFrame
__global__ void BilinearWarpKernel(
    const __half* __restrict__ srcFrame,
    const __half* __restrict__ flowField,
    __half* __restrict__ dstFrame,
    int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int spatialSize = width * height;
    int idx = y * width + x;

    // Fetch flow vectors (dx, dy)
    // channels: flowField[0] = dx, flowField[spatialSize] = dy
    float dx = __half2float(flowField[idx]);
    float dy = __half2float(flowField[spatialSize + idx]);

    // Midpoint interpolation: scale flow displacement by 0.5
    float srcX = static_cast<float>(x) + dx * 0.5f;
    float srcY = static_cast<float>(y) + dy * 0.5f;

    // Bilinear interpolation math
    int x0 = static_cast<int>(floorf(srcX));
    int y0 = static_cast<int>(floorf(srcY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Border clamp bounds
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    float wx1 = srcX - floorf(srcX);
    float wy1 = srcY - floorf(srcY);
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    // Warp all 3 color channels (RGB)
    for (int c = 0; c < 3; ++c) {
        int chOffset = c * spatialSize;

        float p00 = __half2float(srcFrame[chOffset + y0 * width + x0]);
        float p10 = __half2float(srcFrame[chOffset + y0 * width + x1]);
        float p01 = __half2float(srcFrame[chOffset + y1 * width + x0]);
        float p11 = __half2float(srcFrame[chOffset + y1 * width + x1]);

        float val = (p00 * wx0 * wy0) + (p10 * wx1 * wy0) + (p01 * wx0 * wy1) + (p11 * wx1 * wy1);
        dstFrame[chOffset + idx] = __float2half(val);
    }
}

// Host wrapper
void RunBilinearWarp(void* srcFrame, void* flowField, void* dstFrame, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    BilinearWarpKernel<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(srcFrame),
        static_cast<const __half*>(flowField),
        static_cast<__half*>(dstFrame),
        width, height
    );
}
