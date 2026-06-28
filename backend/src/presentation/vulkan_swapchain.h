#pragma once

#include <Windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <cuda_runtime.h>
#include <vector>

class VulkanSwapchain {
public:
    VulkanSwapchain();
    ~VulkanSwapchain();

    // Initialize with target game HWND and source capture dimensions.
    // Creates a separate borderless overlay window for output.
    bool Initialize(HWND targetHwnd, int captureWidth, int captureHeight, int scaleFactor);

    // Call after CUDA has finished writing the upscaled frame to GetPresentSurface().
    void Present();

    void Cleanup();

    // CUDA surface that kernels write the upscaled output into
    cudaExternalMemory_t   GetCudaExternalMemory()  const { return m_CudaExtMem; }
    cudaMipmappedArray_t   GetCudaMappedArray()      const { return m_CudaMipmappedArray; }
    cudaSurfaceObject_t    GetPresentSurface()       const { return m_CudaSurface; }

    // Overlay window management
    HWND GetOverlayHwnd() const { return m_OverlayHwnd; }
    void RepositionOverlay(HWND targetHwnd);

private:
    // Win32 overlay window
    bool CreateOverlayWindow(HWND targetHwnd, int outWidth, int outHeight);

    // Vulkan init steps
    bool CreateInstance();
    bool CreateDeviceAndQueue();
    bool CreateSurface();
    bool CreateSwapchain();
    bool CreateCommandPoolAndBuffers();
    bool CreateSyncObjects();
    bool CreateSharedVulkanImage();
    bool MapVulkanImageToCuda();

    void TransitionImage(VkCommandBuffer cmd, VkImage image,
                         VkImageLayout from, VkImageLayout to,
                         VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                         VkAccessFlags srcAccess, VkAccessFlags dstAccess);

    // Dimensions
    int m_CaptureWidth  = 0;
    int m_CaptureHeight = 0;
    int m_OutWidth      = 0;
    int m_OutHeight     = 0;
    int m_ScaleFactor   = 2;

    // Win32 overlay
    HWND m_OverlayHwnd  = nullptr;
    HINSTANCE m_HInst   = nullptr;

    // Vulkan core
    VkInstance       m_Instance    = VK_NULL_HANDLE;
    VkPhysicalDevice m_PhysDevice  = VK_NULL_HANDLE;
    VkDevice         m_Device      = VK_NULL_HANDLE;
    VkQueue          m_Queue       = VK_NULL_HANDLE;
    uint32_t         m_QueueFamily = 0;

    // Surface & swapchain
    VkSurfaceKHR            m_Surface   = VK_NULL_HANDLE;
    VkSwapchainKHR          m_Swapchain = VK_NULL_HANDLE;
    std::vector<VkImage>    m_SwapImages;
    uint32_t                m_ImageCount = 0;

    // Per-frame command buffers (one per swapchain image)
    VkCommandPool                m_CmdPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_CmdBuffers;

    // Sync: one semaphore pair + fence per frame
    std::vector<VkSemaphore> m_ImageAvailable;
    std::vector<VkSemaphore> m_RenderFinished;
    std::vector<VkFence>     m_InFlightFences;
    uint32_t                 m_FrameIndex = 0;

    // Shared image (CUDA writes here, Vulkan blits to swapchain)
    VkImage        m_SharedImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_SharedMemory = VK_NULL_HANDLE;
    HANDLE         m_SharedHandle = nullptr;
    VkDeviceSize   m_SharedSize   = 0;

    // CUDA interop
    cudaExternalMemory_t  m_CudaExtMem         = nullptr;
    cudaMipmappedArray_t  m_CudaMipmappedArray  = nullptr;
    cudaArray_t           m_CudaArray           = nullptr;
    cudaSurfaceObject_t   m_CudaSurface         = 0;

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
};
