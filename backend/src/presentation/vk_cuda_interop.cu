#include "vk_cuda_interop.h"
#include <iostream>

VkCudaInteropManager::VkCudaInteropManager() {
    m_CudaSemaphore = nullptr;
    m_IsInitialized = false;
}

VkCudaInteropManager::~VkCudaInteropManager() {
    Cleanup();
}

bool VkCudaInteropManager::RegisterTimelineSemaphore(VkDevice vkDevice, VkSemaphore vkSemaphore, HANDLE sharedHandle) {
    if (!vkDevice || !vkSemaphore || !sharedHandle) return false;
    Cleanup();

    // Import the Win32 Shared Handle of Vulkan Semaphore into CUDA
    cudaExternalSemaphoreHandleDesc semDesc = {};
    semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    semDesc.handle.win32.handle = sharedHandle;
    semDesc.flags = 0;

    cudaError_t err = cudaImportExternalSemaphore(&m_CudaSemaphore, &semDesc);
    if (err != cudaSuccess) {
        std::cerr << "[InteropManager] Failed to import external timeline semaphore into CUDA: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    m_IsInitialized = true;
    return true;
}

bool VkCudaInteropManager::SignalSemaphore(cudaStream_t stream, uint64_t value) {
    if (!m_IsInitialized || !m_CudaSemaphore) return false;

    cudaExternalSemaphoreSignalParams sigParams = {};
    sigParams.params.fence.value = value;
    sigParams.flags = 0;

    cudaError_t err = cudaSignalExternalSemaphoresAsync(&m_CudaSemaphore, &sigParams, 1, stream);
    return (err == cudaSuccess);
}

bool VkCudaInteropManager::WaitSemaphore(cudaStream_t stream, uint64_t value) {
    if (!m_IsInitialized || !m_CudaSemaphore) return false;

    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.params.fence.value = value;
    waitParams.flags = 0;

    cudaError_t err = cudaWaitExternalSemaphoresAsync(&m_CudaSemaphore, &waitParams, 1, stream);
    return (err == cudaSuccess);
}

void VkCudaInteropManager::Cleanup() {
    if (m_CudaSemaphore) {
        cudaDestroyExternalSemaphore(m_CudaSemaphore);
        m_CudaSemaphore = nullptr;
    }
    m_IsInitialized = false;
}
