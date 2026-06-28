#pragma once

#include <d3d12.h>
#include <cuda_runtime.h>

class Dx12CudaInterop {
public:
    Dx12CudaInterop();
    ~Dx12CudaInterop();

    bool RegisterSharedResource(ID3D12Resource* d3d12Resource);
    void Cleanup();

    cudaSurfaceObject_t MapAndGetSurface();
    void UnmapSurface();

private:
    ID3D12Resource* m_D3D12Resource = nullptr;
    cudaExternalMemory_t m_CudaExtMemory = nullptr;
    cudaMipmappedArray_t m_CudaMipmappedArray = nullptr;
    cudaArray_t m_CudaArray = nullptr;
    cudaSurfaceObject_t m_CudaSurface = 0;

    int m_Width = 0;
    int m_Height = 0;
    bool m_IsMapped = false;
};
