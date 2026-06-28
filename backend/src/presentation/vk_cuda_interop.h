#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <Windows.h>

class VkCudaInteropManager {
public:
    VkCudaInteropManager();
    ~VkCudaInteropManager();

    // Map a Vulkan sempahore to a CUDA external semaphore for GPU timeline synchronization
    bool RegisterTimelineSemaphore(VkDevice vkDevice, VkSemaphore vkSemaphore, HANDLE sharedHandle);
    
    // Signal the semaphore from CUDA stream
    bool SignalSemaphore(cudaStream_t stream, uint64_t value);
    
    // Wait for the semaphore in CUDA stream
    bool WaitSemaphore(cudaStream_t stream, uint64_t value);

    void Cleanup();

private:
    cudaExternalSemaphore_t m_CudaSemaphore = nullptr;
    bool m_IsInitialized = false;
};
