#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to convert RGB FP16 NCHW Tensor back to BGRA Surface format
// (uint8)
__global__ void BlendOutputKernel(const __half *__restrict__ srcBuffer,
                                  cudaSurfaceObject_t presentSurface, int width,
                                  int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int spatialSize = width * height;
  int idx = y * width + x;

  // Read normalized FP16 color planes
  float r = __half2float(srcBuffer[idx]);
  float g = __half2float(srcBuffer[spatialSize + idx]);
  float b = __half2float(srcBuffer[spatialSize * 2 + idx]);

  // Denormalize and clamp to [0, 255]
  uchar4 pixel;
  pixel.z = static_cast<unsigned char>(max(0.0f, min(255.0f, r * 255.0f))); // R
  pixel.y = static_cast<unsigned char>(max(0.0f, min(255.0f, g * 255.0f))); // G
  pixel.x = static_cast<unsigned char>(max(0.0f, min(255.0f, b * 255.0f))); // B
  pixel.w = 255; // Alpha (opaque)

  // Write to Vulkan present surface
  surf2Dwrite(pixel, presentSurface, x * 4, y);
}

// ── BlendToSurfaceKernel
// ────────────────────────────────────────────────────── Same as BlendOutput
// but writes to an arbitrary BGRA8 surface (used for chaining framegen output
// to upscaler input at original resolution)
__global__ void BlendToSurfaceKernel(const __half *__restrict__ srcBuffer,
                                     cudaSurfaceObject_t outputSurface,
                                     int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int spatialSize = width * height;
  int idx = y * width + x;

  float r = __half2float(srcBuffer[idx]);
  float g = __half2float(srcBuffer[spatialSize + idx]);
  float b = __half2float(srcBuffer[spatialSize * 2 + idx]);

  uchar4 pixel;
  pixel.z = static_cast<unsigned char>(max(0.0f, min(255.0f, r * 255.0f))); // R
  pixel.y = static_cast<unsigned char>(max(0.0f, min(255.0f, g * 255.0f))); // G
  pixel.x = static_cast<unsigned char>(max(0.0f, min(255.0f, b * 255.0f))); // B
  pixel.w = 255; // Alpha (opaque)

  surf2Dwrite(pixel, outputSurface, x * 4, y);
}

// Host wrapper
void RunBlendOutput(void *srcBuffer, cudaSurfaceObject_t presentSurface,
                    int width, int height, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  BlendOutputKernel<<<grid, block, 0, stream>>>(
      static_cast<const __half *>(srcBuffer), presentSurface, width, height);
}

void RunBlendToSurface(void *srcBuffer, cudaSurfaceObject_t outputSurface,
                       int width, int height, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  BlendToSurfaceKernel<<<grid, block, 0, stream>>>(
      static_cast<const __half *>(srcBuffer), outputSurface, width, height);
}
