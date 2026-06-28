#include "dx12_cuda_interop.h"
#include <iostream>

Dx12CudaInterop::Dx12CudaInterop() {
  m_D3D12Resource = nullptr;
  m_CudaExtMemory = nullptr;
  m_CudaMipmappedArray = nullptr;
  m_CudaArray = nullptr;
  m_CudaSurface = 0;
  m_IsMapped = false;
}

Dx12CudaInterop::~Dx12CudaInterop() { Cleanup(); }

bool Dx12CudaInterop::RegisterSharedResource(ID3D12Resource *d3d12Resource) {
  if (!d3d12Resource)
    return false;
  Cleanup();

  m_D3D12Resource = d3d12Resource;
  D3D12_RESOURCE_DESC desc = d3d12Resource->GetDesc();
  m_Width = static_cast<int>(desc.Width);
  m_Height = static_cast<int>(desc.Height);

  // Get parent D3D12 device
  ID3D12Device *device = nullptr;
  HRESULT hr = d3d12Resource->GetDevice(IID_PPV_ARGS(&device));
  if (FAILED(hr) || !device)
    return false;

  // Create shared handle
  HANDLE sharedHandle = nullptr;
  hr = device->CreateSharedHandle(d3d12Resource, nullptr, GENERIC_ALL, nullptr,
                                  &sharedHandle);
  if (FAILED(hr) || !sharedHandle) {
    std::cerr << "[Interop] Failed to get shared handle for DX12 Resource."
              << std::endl;
    if (device)
      device->Release();
    return false;
  }

  D3D12_RESOURCE_ALLOCATION_INFO allocInfo =
      device->GetResourceAllocationInfo(0, 1, &desc);
  device->Release();
  UINT64 resourceSize = allocInfo.SizeInBytes;

  // CUDA import setup — use the Win32 shared HANDLE
  cudaExternalMemoryHandleDesc extMemDesc = {};
  extMemDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
  extMemDesc.handle.win32.handle = sharedHandle;
  extMemDesc.size = resourceSize;
  extMemDesc.flags = cudaExternalMemoryDedicated;

  cudaError_t err = cudaImportExternalMemory(&m_CudaExtMemory, &extMemDesc);
  CloseHandle(sharedHandle); // Handle can be closed once imported

  if (err != cudaSuccess) {
    std::cerr << "[Interop] cudaImportExternalMemory failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Map mipmapped array from external memory
  // B8G8R8A8_UNORM: 4 channels × 8-bit unsigned
  cudaExternalMemoryMipmappedArrayDesc mipDesc = {};
  mipDesc.offset = 0;
  mipDesc.formatDesc.x = 8; // B channel (8-bit)
  mipDesc.formatDesc.y = 8; // G channel (8-bit)
  mipDesc.formatDesc.z = 8; // R channel (8-bit)
  mipDesc.formatDesc.w = 8; // A channel (8-bit)
  mipDesc.formatDesc.f = cudaChannelFormatKindUnsigned;
  mipDesc.extent.width = m_Width;
  mipDesc.extent.height = m_Height;
  mipDesc.extent.depth = 0; // 2D array
  mipDesc.numLevels = 1;
  mipDesc.flags =
      cudaArraySurfaceLoadStore; // CRITICAL: enables surface read/write

  err = cudaExternalMemoryGetMappedMipmappedArray(&m_CudaMipmappedArray,
                                                  m_CudaExtMemory, &mipDesc);
  if (err != cudaSuccess) {
    std::cerr << "[Interop] cudaExternalMemoryGetMappedMipmappedArray failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Get actual level 0 array
  err = cudaGetMipmappedArrayLevel(&m_CudaArray, m_CudaMipmappedArray, 0);
  if (err != cudaSuccess)
    return false;

  // Create CUDA surface object for kernel read/write
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_CudaArray;

  err = cudaCreateSurfaceObject(&m_CudaSurface, &resDesc);
  if (err != cudaSuccess) {
    std::cerr << "[Interop] cudaCreateSurfaceObject failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  std::cout << "[Interop] Registered and mapped DX12 resource: " << m_Width
            << "x" << m_Height << " to CUDA Surface." << std::endl;
  return true;
}

cudaSurfaceObject_t Dx12CudaInterop::MapAndGetSurface() {
  // Windows graphics memory is already bound to GPU, but we flag it as mapped
  m_IsMapped = true;
  return m_CudaSurface;
}

void Dx12CudaInterop::UnmapSurface() { m_IsMapped = false; }

void Dx12CudaInterop::Cleanup() {
  if (m_CudaSurface) {
    cudaDestroySurfaceObject(m_CudaSurface);
    m_CudaSurface = 0;
  }
  if (m_CudaMipmappedArray) {
    cudaFreeMipmappedArray(m_CudaMipmappedArray);
    m_CudaMipmappedArray = nullptr;
  }
  if (m_CudaExtMemory) {
    cudaDestroyExternalMemory(m_CudaExtMemory);
    m_CudaExtMemory = nullptr;
  }
  m_D3D12Resource = nullptr;
  m_IsMapped = false;
}