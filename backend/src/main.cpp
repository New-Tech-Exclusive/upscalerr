#include <Windows.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "capture/dx12_cuda_interop.h"
#include "capture/wgc_capture.h"
#include "inference/framegen_pipeline.h"
#include "inference/upscale_pipeline.h"
#include "ipc/pipe_server.h"
#include "ipc/protocol.h"
#include "presentation/vulkan_swapchain.h"

// Global control flags
std::atomic<bool> g_Running(true);
std::atomic<bool> g_UpscaleEnabled(true);
std::atomic<bool> g_FrameGenEnabled(true);
std::atomic<int> g_ScaleFactor(2);
std::atomic<HWND> g_TargetWindow(nullptr);

// Performance stats
std::atomic<float> g_FPS(0.0f);
std::atomic<float> g_Latency(0.0f);

// Named Pipe Server callback
void OnIpcMessage(const IpcCommand &cmd, IpcResponse &response) {
  switch (cmd.type) {
  case CMD_TOGGLE_UP:
    g_UpscaleEnabled = (cmd.payload.toggleValue != 0);
    response.status = RESP_OK;
    std::cout << "[IPC] Toggle Upscale: " << (g_UpscaleEnabled ? "ON" : "OFF")
              << std::endl;
    break;
  case CMD_TOGGLE_FG:
    g_FrameGenEnabled = (cmd.payload.toggleValue != 0);
    response.status = RESP_OK;
    std::cout << "[IPC] Toggle FrameGen: " << (g_FrameGenEnabled ? "ON" : "OFF")
              << std::endl;
    break;
  case CMD_SET_SCALE:
    if (cmd.payload.scaleFactor == 2 || cmd.payload.scaleFactor == 3 ||
        cmd.payload.scaleFactor == 4) {
      g_ScaleFactor = cmd.payload.scaleFactor;
      response.status = RESP_OK;
      std::cout << "[IPC] Set Scale Factor: " << g_ScaleFactor << "x"
                << std::endl;
    } else {
      response.status = RESP_ERROR;
    }
    break;
  case CMD_SET_TARGET:
    g_TargetWindow = reinterpret_cast<HWND>(cmd.payload.windowHandle);
    response.status = RESP_OK;
    std::cout << "[IPC] Set Target Window HWND: " << g_TargetWindow.load()
              << std::endl;
    break;
  case CMD_GET_STATS:
    response.status = RESP_OK;
    response.payload.stats.fps = g_FPS.load();
    response.payload.stats.latency_ms = g_Latency.load();
    break;
  case CMD_SHUTDOWN:
    g_Running = false;
    response.status = RESP_OK;
    std::cout << "[IPC] Shutdown command received" << std::endl;
    break;
  default:
    response.status = RESP_ERROR;
    break;
  }
}

int main() {
  std::cout << "=========================================================="
            << std::endl;
  std::cout << "   Upscalerr Backend - Initializing NVIDIA Pipeline"
            << std::endl;
  std::cout << "=========================================================="
            << std::endl;

  // Start named pipe server on dedicated thread
  PipeServer ipcServer(L"\\\\.\\pipe\\upscalerr", OnIpcMessage);
  if (!ipcServer.Start()) {
    std::cerr << "[FATAL] Failed to start IPC Named Pipe Server." << std::endl;
    return -1;
  }
  std::cout << "[IPC] Named Pipe Server listening..." << std::endl;

  // Initialize D3D12 device and Graphics Capture
  std::unique_ptr<WgcCapture> capture = std::make_unique<WgcCapture>();
  std::unique_ptr<Dx12CudaInterop> dx12Cuda =
      std::make_unique<Dx12CudaInterop>();

  // Initialize presentation overlay window & Vulkan swapchain
  std::unique_ptr<VulkanSwapchain> swapchain =
      std::make_unique<VulkanSwapchain>();

  // CUDA inference engines
  std::unique_ptr<UpscalePipeline> upscalePipeline =
      std::make_unique<UpscalePipeline>();
  std::unique_ptr<FrameGenPipeline> frameGenPipeline =
      std::make_unique<FrameGenPipeline>();

  bool pipelineInitialized = false;
  HWND currentHwnd = nullptr;

  // Intermediate BGRA8 surface for framegen output (at original capture res)
  // Used to chain framegen → upscale when both are enabled
  cudaSurfaceObject_t m_FrameGenOutputSurface = 0;
  cudaArray_t m_FrameGenOutputArray = nullptr;
  size_t m_FrameGenOutputSize = 0;

  auto lastTime = std::chrono::high_resolution_clock::now();
  int frameCount = 0;

  // Main Engine Processing Loop
  while (g_Running) {
    // Handle target window changes
    HWND targetHwnd = g_TargetWindow.load();
    if (targetHwnd != currentHwnd) {
      if (targetHwnd && IsWindow(targetHwnd)) {
        std::cout << "[MAIN] Target window changed. Binding capture to HWND: "
                  << targetHwnd << std::endl;

        // Stop any existing capture
        capture->Stop();
        pipelineInitialized = false;

        // Clean up any intermediate surfaces
        if (m_FrameGenOutputSurface) {
          cudaDestroySurfaceObject(m_FrameGenOutputSurface);
          m_FrameGenOutputSurface = 0;
        }
        if (m_FrameGenOutputArray) {
          cudaFreeArray(m_FrameGenOutputArray);
          m_FrameGenOutputArray = nullptr;
        }

        // Init new capture session
        if (capture->Start(targetHwnd)) {
          currentHwnd = targetHwnd;

          // Get target dimensions
          RECT rect;
          GetClientRect(currentHwnd, &rect);
          int width = rect.right - rect.left;
          int height = rect.bottom - rect.top;

          // Ensure dimensions are positive
          if (width > 0 && height > 0) {
            std::cout << "[MAIN] Capture window dimensions: " << width << "x"
                      << height << std::endl;

            // Allocate intermediate BGRA8 surface at capture resolution
            // for framegen→upscale chaining
            {
              cudaChannelFormatDesc channelDesc =
                  cudaCreateChannelDesc<uchar4>();
              cudaError_t err =
                  cudaMallocArray(&m_FrameGenOutputArray, &channelDesc, width,
                                  height, cudaArraySurfaceLoadStore);
              if (err != cudaSuccess) {
                std::cerr << "[MAIN] Failed to allocate intermediate FG output "
                             "array: "
                          << cudaGetErrorString(err) << std::endl;
              } else {
                cudaResourceDesc resDesc = {};
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = m_FrameGenOutputArray;
                err =
                    cudaCreateSurfaceObject(&m_FrameGenOutputSurface, &resDesc);
                if (err != cudaSuccess) {
                  std::cerr
                      << "[MAIN] Failed to create FG output surface object"
                      << std::endl;
                  cudaFreeArray(m_FrameGenOutputArray);
                  m_FrameGenOutputArray = nullptr;
                } else {
                  m_FrameGenOutputSize = width * height * 4;
                  std::cout
                      << "[MAIN] Allocated intermediate FG→Upscale surface: "
                      << width << "x" << height << std::endl;
                }
              }
            }

            // Setup D3D12-CUDA texture mappings
            ID3D12Resource *d3d12Res = capture->GetSharedResource();
            if (!d3d12Res) {
              std::cerr
                  << "[MAIN] ERROR: capture->GetSharedResource() returned null!"
                  << std::endl;
            } else if (!dx12Cuda->RegisterSharedResource(d3d12Res)) {
              std::cerr << "[MAIN] ERROR: "
                           "Dx12CudaInterop::RegisterSharedResource failed!"
                        << std::endl;
            } else if (!swapchain->Initialize(currentHwnd, width, height,
                                              g_ScaleFactor.load())) {
              std::cerr << "[MAIN] ERROR: VulkanSwapchain::Initialize failed!"
                        << std::endl;
            } else if (!upscalePipeline->Initialize(g_ScaleFactor.load(), width,
                                                    height)) {
              std::cerr << "[MAIN] ERROR: UpscalePipeline::Initialize failed!"
                        << std::endl;
            } else if (!frameGenPipeline->Initialize(width, height)) {
              std::cerr << "[MAIN] ERROR: FrameGenPipeline::Initialize failed!"
                        << std::endl;
            } else {
              pipelineInitialized = true;
              std::cout << "[MAIN] Real-time hardware pipelines mapped and "
                           "validated!"
                        << std::endl;
            }
          }
        }
      } else {
        capture->Stop();
        currentHwnd = nullptr;
        pipelineInitialized = false;
        swapchain->Cleanup();
        upscalePipeline->Cleanup();
        frameGenPipeline->Cleanup();

        if (m_FrameGenOutputSurface) {
          cudaDestroySurfaceObject(m_FrameGenOutputSurface);
          m_FrameGenOutputSurface = 0;
        }
        if (m_FrameGenOutputArray) {
          cudaFreeArray(m_FrameGenOutputArray);
          m_FrameGenOutputArray = nullptr;
        }
      }
    }

    if (!pipelineInitialized || !currentHwnd) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto frameStart = std::chrono::high_resolution_clock::now();

    // 1. Capture frame N (Zero-copy GPU texture handle acquisition)
    if (!capture->AcquireNextFrame()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // 2. Map captured DX12 texture into CUDA array
    cudaSurfaceObject_t rawInputSurface = dx12Cuda->MapAndGetSurface();
    if (!rawInputSurface) {
      capture->ReleaseFrame();
      continue;
    }

    // 3. Process spatial upscaler and optical flow frame generation
    cudaSurfaceObject_t vkPresentSurface = swapchain->GetPresentSurface();

    if (g_FrameGenEnabled && g_UpscaleEnabled) {
      // FULL PIPELINE: FrameGen → Upscale (chained at full resolution)
      //
      // Run framegen first at original res, writing output to intermediate
      // BGRA8 surface, then upscale the interpolated frame to final output.
      //
      // FrameGen internally maintains PrevFrameBuffer/CurrFrameBuffer state,
      // warps the previous frame, and writes the BGRA8 result to our
      // intermediate surface. The upscaler reads this intermediate surface
      // as if it were a raw capture frame and produces the final upscaled
      // output on the Vulkan present surface.
      frameGenPipeline->ProcessToSurface(rawInputSurface,
                                         m_FrameGenOutputSurface);
      upscalePipeline->Process(m_FrameGenOutputSurface, vkPresentSurface);
    } else if (g_FrameGenEnabled) {
      // Frame generation only (no upscaling): output goes directly to present
      frameGenPipeline->ProcessAndWarp(rawInputSurface, vkPresentSurface);
    } else if (g_UpscaleEnabled) {
      // Spatial AI upscale only: raw capture → direct to Vulkan mapped memory
      upscalePipeline->Process(rawInputSurface, vkPresentSurface);
    } else {
      // Passthrough: just copy/color-convert without processing
      upscalePipeline->CopyDirect(rawInputSurface, vkPresentSurface);
    }

    // 4. Unmap resources and present
    dx12Cuda->UnmapSurface();
    capture->ReleaseFrame();

    swapchain->Present();

    auto frameEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = frameEnd - frameStart;
    g_Latency = duration.count();

    // Stats tracking
    frameCount++;
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = now - lastTime;
    if (elapsed.count() >= 1.0f) {
      g_FPS = static_cast<float>(frameCount) / elapsed.count();
      frameCount = 0;
      lastTime = now;
    }
  }

  // Clean up pipelines
  if (m_FrameGenOutputSurface) {
    cudaDestroySurfaceObject(m_FrameGenOutputSurface);
  }
  if (m_FrameGenOutputArray) {
    cudaFreeArray(m_FrameGenOutputArray);
  }

  capture->Stop();
  dx12Cuda->Cleanup();
  upscalePipeline->Cleanup();
  frameGenPipeline->Cleanup();
  swapchain->Cleanup();
  ipcServer.Stop();

  std::cout << "[MAIN] Backend exited gracefully." << std::endl;
  return 0;
}