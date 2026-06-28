#include "upscale_pipeline.h"
#include <filesystem>
#include <iostream>

// Declarations of custom CUDA kernels defined in kernels/ directory
extern void RunColorConvert(cudaSurfaceObject_t rawSurface, void *dstBuffer,
                            int width, int height, cudaStream_t stream);
extern void RunBlendOutput(void *srcBuffer, cudaSurfaceObject_t presentSurface,
                           int width, int height, cudaStream_t stream);
extern void RunCopyDirect(cudaSurfaceObject_t rawSurface,
                          cudaSurfaceObject_t presentSurface, int width,
                          int height, cudaStream_t stream);

UpscalePipeline::UpscalePipeline() {
  m_TrtEngine = std::make_unique<TrtEngine>();
  m_Stream = nullptr;
}

UpscalePipeline::~UpscalePipeline() { Cleanup(); }

bool UpscalePipeline::Initialize(int scaleFactor, int inputWidth,
                                 int inputHeight) {
  Cleanup();

  m_ScaleFactor = scaleFactor;
  m_InputWidth = inputWidth;
  m_InputHeight = inputHeight;
  m_OutputWidth = inputWidth * scaleFactor;
  m_OutputHeight = inputHeight * scaleFactor;

  // Create stream
  cudaError_t err = cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking);
  if (err != cudaSuccess)
    return false;

  // Resolve engine path
  std::string engineName =
      "profiles/espcn_" + std::to_string(scaleFactor) + "x_fp16.engine";
  if (!m_TrtEngine->Load(engineName)) {
    // Fallback check in local directory
    engineName = "espcn_" + std::to_string(scaleFactor) + "x_fp16.engine";
    if (!m_TrtEngine->Load(engineName)) {
      std::cerr
          << "[UpscalePipeline] Failed to find or load ESPCN Engine profile."
          << std::endl;
      return false;
    }
  }

  // Bind dynamic sizes
  if (!m_TrtEngine->SetInputDimensions("input", 1, 3, m_InputHeight,
                                       m_InputWidth)) {
    return false;
  }

  // Allocate memory buffers
  if (!m_TrtEngine->AllocateBuffers()) {
    return false;
  }

  m_InputDevBuffer = m_TrtEngine->GetInputBuffer("input");
  m_OutputDevBuffer = m_TrtEngine->GetOutputBuffer("output");

  if (!m_InputDevBuffer || !m_OutputDevBuffer) {
    std::cerr
        << "[UpscalePipeline] Input/Output tensor binding allocation failed."
        << std::endl;
    return false;
  }

  // Initialize CUDA Graphs for sub-2ms executions
  if (!m_TrtEngine->InitializeGraph(m_Stream)) {
    std::cerr << "[UpscalePipeline] Failed to capture CUDA Graph. Defaulting "
                 "to stream enqueue."
              << std::endl;
  }

  std::cout << "[UpscalePipeline] Initialized successfully. Scale: "
            << m_ScaleFactor << "x, Dims: " << m_InputWidth << "x"
            << m_InputHeight << " -> " << m_OutputWidth << "x" << m_OutputHeight
            << std::endl;
  return true;
}

bool UpscalePipeline::Process(cudaSurfaceObject_t rawSurface,
                              cudaSurfaceObject_t presentSurface) {
  if (!m_TrtEngine || !m_Stream)
    return false;

  // 1. Run Preprocessing: BGRA Surface -> RGB FP16 NCHW Tensor
  RunColorConvert(rawSurface, m_InputDevBuffer, m_InputWidth, m_InputHeight,
                  m_Stream);

  // 2. Run TRT inference (via CUDA Graph if initialized, otherwise enqueue)
  if (!m_TrtEngine->LaunchGraph(m_Stream)) {
    if (!m_TrtEngine->Enqueue(m_Stream)) {
      return false;
    }
  }

  // 3. Run Postprocessing: RGB FP16 NCHW -> BGRA Vulkan Surface
  RunBlendOutput(m_OutputDevBuffer, presentSurface, m_OutputWidth,
                 m_OutputHeight, m_Stream);

  // Synchronize current stream to complete operations
  cudaStreamSynchronize(m_Stream);
  return true;
}

bool UpscalePipeline::CopyDirect(cudaSurfaceObject_t rawSurface,
                                 cudaSurfaceObject_t presentSurface) {
  if (!m_Stream)
    return false;

  // Fallback: fast GPU surface copy without upscaling
  RunCopyDirect(rawSurface, presentSurface, m_InputWidth, m_InputHeight,
                m_Stream);
  cudaStreamSynchronize(m_Stream);
  return true;
}

void QSync(cudaStream_t stream) {
  if (stream)
    cudaStreamSynchronize(stream);
}

void UpscalePipeline::Cleanup() {
  if (m_Stream) {
    cudaStreamDestroy(m_Stream);
    m_Stream = nullptr;
  }
  if (m_TrtEngine) {
    m_TrtEngine->Cleanup();
  }
  m_InputDevBuffer = nullptr;
  m_OutputDevBuffer = nullptr;
}
