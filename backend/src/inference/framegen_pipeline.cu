#include "framegen_pipeline.h"
#include <iostream>

// Kernel declarations
extern void RunColorConvert(cudaSurfaceObject_t rawSurface, void *dstBuffer,
                            int width, int height, cudaStream_t stream);
extern void RunConcatFrames(void *framePrev, void *frameNext, void *dstConcat,
                            int width, int height, cudaStream_t stream);
extern void RunBilinearWarp(void *srcFrame, void *flowField, void *dstFrame,
                            int width, int height, cudaStream_t stream);
extern void RunBlendOutput(void *srcBuffer, cudaSurfaceObject_t presentSurface,
                           int width, int height, cudaStream_t stream);
extern void RunBlendToSurface(void *srcBuffer,
                              cudaSurfaceObject_t outputSurface, int width,
                              int height, cudaStream_t stream);

FrameGenPipeline::FrameGenPipeline() {
  m_TrtEngine = std::make_unique<TrtEngine>();
  m_Stream = nullptr;
  m_HasPrevFrame = false;
}

FrameGenPipeline::~FrameGenPipeline() { Cleanup(); }

bool FrameGenPipeline::Initialize(int width, int height) {
  Cleanup();

  m_Width = width;
  m_Height = height;

  cudaError_t err = cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking);
  if (err != cudaSuccess)
    return false;

  // Load FlowNet engine
  std::string engineName = "profiles/flownet_fp16.engine";
  if (!m_TrtEngine->Load(engineName)) {
    engineName = "flownet_fp16.engine";
    if (!m_TrtEngine->Load(engineName)) {
      std::cerr
          << "[FrameGenPipeline] Failed to find or load FlowNet Engine profile."
          << std::endl;
      return false;
    }
  }

  // Set input dims: 6 channels for concat frame N-1 and N
  if (!m_TrtEngine->SetInputDimensions("frames_concat", 1, 6, m_Height,
                                       m_Width)) {
    return false;
  }

  if (!m_TrtEngine->AllocateBuffers()) {
    return false;
  }

  // Get TRT IO buffers
  m_InputConcatBuffer = m_TrtEngine->GetInputBuffer("frames_concat");
  m_OutputFlowBuffer = m_TrtEngine->GetOutputBuffer("flow");

  if (!m_InputConcatBuffer || !m_OutputFlowBuffer) {
    std::cerr << "[FrameGenPipeline] FlowNet IO binding allocation failed."
              << std::endl;
    return false;
  }

  // Allocate ring buffers for state tracking [1, 3, H, W] in FP16 (2
  // bytes/pixel)
  size_t frameBytes = m_Width * m_Height * 3 * sizeof(unsigned short);
  err = cudaMalloc(&m_PrevFrameBuffer, frameBytes);
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&m_CurrFrameBuffer, frameBytes);
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&m_WarpedFrameBuffer, frameBytes);
  if (err != cudaSuccess)
    return false;

  cudaMemset(m_PrevFrameBuffer, 0, frameBytes);
  cudaMemset(m_CurrFrameBuffer, 0, frameBytes);
  cudaMemset(m_WarpedFrameBuffer, 0, frameBytes);

  m_HasPrevFrame = false;

  // Capture CUDA Graph
  if (!m_TrtEngine->InitializeGraph(m_Stream)) {
    std::cerr << "[FrameGenPipeline] Failed to capture CUDA Graph. Defaulting "
                 "to stream enqueue."
              << std::endl;
  }

  std::cout << "[FrameGenPipeline] Initialized successfully. Resolution: "
            << m_Width << "x" << m_Height << std::endl;
  return true;
}

bool FrameGenPipeline::ProcessAndWarp(cudaSurfaceObject_t rawSurface,
                                      cudaSurfaceObject_t presentSurface) {
  if (!m_Stream || !m_TrtEngine)
    return false;

  // 1. Preprocess new frame into CurrFrameBuffer
  RunColorConvert(rawSurface, m_CurrFrameBuffer, m_Width, m_Height, m_Stream);

  if (!m_HasPrevFrame) {
    // First frame logic: copy current to previous and skip flow interpolation
    cudaMemcpyAsync(m_PrevFrameBuffer, m_CurrFrameBuffer,
                    m_Width * m_Height * 3 * sizeof(unsigned short),
                    cudaMemcpyDeviceToDevice, m_Stream);
    m_HasPrevFrame = true;
    cudaStreamSynchronize(m_Stream);
    return true;
  }

  // 2. Concatenate PrevFrameBuffer (N-1) and CurrFrameBuffer (N) to feed model
  RunConcatFrames(m_PrevFrameBuffer, m_CurrFrameBuffer, m_InputConcatBuffer,
                  m_Width, m_Height, m_Stream);

  // 3. Compute optical flow
  if (!m_TrtEngine->LaunchGraph(m_Stream)) {
    if (!m_TrtEngine->Enqueue(m_Stream)) {
      return false;
    }
  }

  // 4. Warp previous frame using computed motion vectors
  RunBilinearWarp(m_PrevFrameBuffer, m_OutputFlowBuffer, m_WarpedFrameBuffer,
                  m_Width, m_Height, m_Stream);

  // 5. Present the generated intermediate frame
  RunBlendOutput(m_WarpedFrameBuffer, presentSurface, m_Width, m_Height,
                 m_Stream);

  // Swap states
  void *temp = m_PrevFrameBuffer;
  m_PrevFrameBuffer = m_CurrFrameBuffer;
  m_CurrFrameBuffer = temp;

  cudaStreamSynchronize(m_Stream);
  return true;
}

bool FrameGenPipeline::ProcessToSurface(cudaSurfaceObject_t rawSurface,
                                        cudaSurfaceObject_t outputSurface) {
  if (!m_Stream || !m_TrtEngine)
    return false;

  // 1. Preprocess new frame into CurrFrameBuffer
  RunColorConvert(rawSurface, m_CurrFrameBuffer, m_Width, m_Height, m_Stream);

  if (!m_HasPrevFrame) {
    // First frame logic: copy current to previous, passthrough raw input to
    // outputSurface
    cudaMemcpyAsync(m_PrevFrameBuffer, m_CurrFrameBuffer,
                    m_Width * m_Height * 3 * sizeof(unsigned short),
                    cudaMemcpyDeviceToDevice, m_Stream);
    // Fall through: write raw surface to output as-is for the first frame
    // (we use CopyDirect kernel on the outputSurface)
    RunBlendOutput(m_CurrFrameBuffer, outputSurface, m_Width, m_Height,
                   m_Stream);
    m_HasPrevFrame = true;
    cudaStreamSynchronize(m_Stream);
    return true;
  }

  // 2. Concatenate PrevFrameBuffer (N-1) and CurrFrameBuffer (N) to feed model
  RunConcatFrames(m_PrevFrameBuffer, m_CurrFrameBuffer, m_InputConcatBuffer,
                  m_Width, m_Height, m_Stream);

  // 3. Compute optical flow
  if (!m_TrtEngine->LaunchGraph(m_Stream)) {
    if (!m_TrtEngine->Enqueue(m_Stream)) {
      return false;
    }
  }

  // 4. Warp previous frame using computed motion vectors
  RunBilinearWarp(m_PrevFrameBuffer, m_OutputFlowBuffer, m_WarpedFrameBuffer,
                  m_Width, m_Height, m_Stream);

  // 5. Write the warped frame (RGB FP16) to the intermediate BGRA8 surface
  //    (for chaining to the upscaler)
  RunBlendToSurface(m_WarpedFrameBuffer, outputSurface, m_Width, m_Height,
                    m_Stream);

  // Swap states
  void *temp = m_PrevFrameBuffer;
  m_PrevFrameBuffer = m_CurrFrameBuffer;
  m_CurrFrameBuffer = temp;

  cudaStreamSynchronize(m_Stream);
  return true;
}

void FrameGenPipeline::Cleanup() {
  if (m_Stream) {
    cudaStreamDestroy(m_Stream);
    m_Stream = nullptr;
  }
  if (m_TrtEngine) {
    m_TrtEngine->Cleanup();
  }
  if (m_PrevFrameBuffer) {
    cudaFree(m_PrevFrameBuffer);
    m_PrevFrameBuffer = nullptr;
  }
  if (m_CurrFrameBuffer) {
    cudaFree(m_CurrFrameBuffer);
    m_CurrFrameBuffer = nullptr;
  }
  if (m_WarpedFrameBuffer) {
    cudaFree(m_WarpedFrameBuffer);
    m_WarpedFrameBuffer = nullptr;
  }
  m_InputConcatBuffer = nullptr;
  m_OutputFlowBuffer = nullptr;
  m_HasPrevFrame = false;
}
