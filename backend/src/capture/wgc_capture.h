#pragma once

#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.System.h>

class WgcCapture {
public:
    WgcCapture();
    ~WgcCapture();

    bool Start(HWND targetHwnd);
    void Stop();
    
    bool AcquireNextFrame();
    void ReleaseFrame();

    ID3D12Resource* GetSharedResource() const { return m_D3D12SharedResource.Get(); }
    ID3D11Texture2D* GetD3D11CapturedTexture() const { return m_D3D11CapturedTexture.Get(); }

private:
    bool InitializeD3DDevices();
    bool SetupCaptureItem(HWND targetHwnd);
    bool CreateFramePool();

    Microsoft::WRL::ComPtr<ID3D11Device> m_D3D11Device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_D3D11Context;
    Microsoft::WRL::ComPtr<ID3D12Device> m_D3D12Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_D3D12Queue;

    // Direct3D 11-on-12 wrapped resource to share textures
    Microsoft::WRL::ComPtr<ID3D12Resource> m_D3D12SharedResource;
    HANDLE m_SharedResourceHandle = nullptr;

    // WinRT Windows Graphics Capture variables
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_CaptureItem = nullptr;
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_FramePool = nullptr;
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_CaptureSession = nullptr;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_WinrtDevice = nullptr;

    // Captured frame details
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11CapturedTexture;
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame m_CurrentFrame = nullptr;
    winrt::Windows::System::DispatcherQueueController m_DispatcherQueueController = nullptr;

    bool m_IsCapturing = false;
    int m_Width = 0;
    int m_Height = 0;
};
