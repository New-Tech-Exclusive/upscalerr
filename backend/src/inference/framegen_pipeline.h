#pragma once

#include "trt_engine.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>

class FrameGenPipeline {
public:
  FrameGenPipeline();
  ~FrameGenPipeline();

  bool Initialize(int width, int height);
  void Cleanup();

  // Feeds next captured surface, extracts optical flow vectors, warps, and
  // blends the intermediate frame.
  bool ProcessAndWarp(cudaSurfaceObject_t rawSurface,
                      cudaSurfaceObject_t presentSurface);

  // Feeds next captured surface, writes the warped BGRA8 result to
  // outputSurface (for chaining framegen output to the upscaler at original
  // resolution)
  bool ProcessToSurface(cudaSurfaceObject_t rawSurface,
                        cudaSurfaceObject_t outputSurface);

private:
  std::unique_ptr<TrtEngine> m_TrtEngine;
  cudaStream_t m_Stream = nullptr;

  int m_Width = 0;
  int m_Height = 0;

  // Buffer to save previous frame N-1 RGB FP16 tensor
  void *m_PrevFrameBuffer = nullptr;
  // Buffer to save current frame N RGB FP16 tensor
  void *m_CurrFrameBuffer = nullptr;

  // Concatenated frames buffer [1, 6, H, W] to feed TRT
  void *m_InputConcatBuffer = nullptr;

  // Output buffer pointers owned/managed by TrtEngine
  void *m_OutputFlowBuffer = nullptr;  // flow [1, 2, H, W]
  void *m_WarpedFrameBuffer = nullptr; // warped frame RGB FP16 [1, 3, H, W]

  bool m_HasPrevFrame = false;
};