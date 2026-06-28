#include "wgc_capture.h"
#include <iostream>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <Windows.Graphics.Capture.Interop.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.Foundation.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <d3d11on12.h>
#include <DispatcherQueue.h>

// Helper to create DispatcherQueueController
winrt::Windows::System::DispatcherQueueController CreateDispatcherQueueController() {
    DispatcherQueueOptions options = {
        sizeof(DispatcherQueueOptions),
        DQTYPE_THREAD_CURRENT,
        DQTAT_COM_NONE
    };

    ABI::Windows::System::IDispatcherQueueController* controller = nullptr;
    winrt::check_hresult(CreateDispatcherQueueController(options, &controller));
    
    winrt::Windows::System::DispatcherQueueController winrtController{ nullptr };
    winrt::attach_abi(winrtController, controller);
    return winrtController;
}

WgcCapture::WgcCapture() {
    // Initialize COM for WinRT
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    m_DispatcherQueueController = CreateDispatcherQueueController();
}

WgcCapture::~WgcCapture() {
    Stop();
}

bool WgcCapture::InitializeD3DDevices() {
    // 1. Create DX12 Device and Command Queue
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_D3D12Device));
    if (FAILED(hr)) {
        std::cerr << "[WGC] Failed to create D3D12 Device." << std::endl;
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = m_D3D12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_D3D12Queue));
    if (FAILED(hr)) return false;

    // 2. Create D3D11On12 Device
    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> d3d11Context;
    Microsoft::WRL::ComPtr<ID3D11On12Device> d3d11On12Device;

    IUnknown* queueUnknown = m_D3D12Queue.Get();
    hr = D3D11On12CreateDevice(
        m_D3D12Device.Get(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr, 0,
        &queueUnknown, 1,
        0,
        &d3d11Device,
        &d3d11Context,
        nullptr
    );
    if (FAILED(hr)) {
        std::cerr << "[WGC] Failed to create D3D11On12 Device." << std::endl;
        return false;
    }

    m_D3D11Device = d3d11Device;
    m_D3D11Context = d3d11Context;

    // 3. Create WinRT Direct3D Device for Capture APIs
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    hr = m_D3D11Device.As(&dxgiDevice);
    if (FAILED(hr)) return false;

    winrt::com_ptr<::IInspectable> inspectableDevice;
    hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectableDevice.put());
    if (FAILED(hr)) return false;

    m_WinrtDevice = inspectableDevice.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
    return true;
}

bool WgcCapture::SetupCaptureItem(HWND targetHwnd) {
    auto interop_factory = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    HRESULT hr = interop_factory->CreateForWindow(
        targetHwnd,
        winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
        winrt::put_abi(item)
    );

    if (FAILED(hr)) {
        std::cerr << "[WGC] Failed to create GraphicsCaptureItem for HWND: " << targetHwnd << std::endl;
        return false;
    }

    m_CaptureItem = item;
    winrt::Windows::Graphics::SizeInt32 size = m_CaptureItem.Size();
    m_Width = size.Width;
    m_Height = size.Height;
    return true;
}

bool WgcCapture::CreateFramePool() {
    // Create DXGI/D3D12 shared resource for our output mappings
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = m_Width;
    desc.Height = m_Height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = m_D3D12Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_D3D12SharedResource)
    );
    if (FAILED(hr)) {
        std::cerr << "[WGC] Failed to create D3D12 shared resource." << std::endl;
        return false;
    }

    // Export shared handle
    hr = m_D3D12Device->CreateSharedHandle(
        m_D3D12SharedResource.Get(),
        nullptr,
        GENERIC_ALL,
        nullptr,
        &m_SharedResourceHandle
    );
    if (FAILED(hr)) return false;

    // Create frame pool using the WinRT Direct3D Device
    m_FramePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
        m_WinrtDevice,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        m_CaptureItem.Size()
    );

    return true;
}

bool WgcCapture::Start(HWND targetHwnd) {
    if (m_IsCapturing) Stop();

    if (!InitializeD3DDevices()) return false;
    if (!SetupCaptureItem(targetHwnd)) return false;
    if (!CreateFramePool()) return false;

    m_CaptureSession = m_FramePool.CreateCaptureSession(m_CaptureItem);
    m_CaptureSession.IsBorderRequired(false);
    m_CaptureSession.IsCursorCaptureEnabled(false);
    
    m_CaptureSession.StartCapture();
    m_IsCapturing = true;
    std::cout << "[WGC] Capture Session started successfully." << std::endl;
    return true;
}

void WgcCapture::Stop() {
    if (!m_IsCapturing) return;

    if (m_CaptureSession) {
        m_CaptureSession.Close();
        m_CaptureSession = nullptr;
    }
    if (m_FramePool) {
        m_FramePool.Close();
        m_FramePool = nullptr;
    }
    m_CaptureItem = nullptr;
    m_D3D11CapturedTexture = nullptr;
    m_D3D12SharedResource = nullptr;

    if (m_SharedResourceHandle) {
        CloseHandle(m_SharedResourceHandle);
        m_SharedResourceHandle = nullptr;
    }

    m_IsCapturing = false;
    std::cout << "[WGC] Capture Session stopped." << std::endl;
}

bool WgcCapture::AcquireNextFrame() {
    if (!m_IsCapturing || !m_FramePool) return false;

    // Try to pull a frame
    m_CurrentFrame = m_FramePool.TryGetNextFrame();
    if (!m_CurrentFrame) return false;

    // Pull texture out of the capture frame
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface surface = m_CurrentFrame.Surface();
    auto surfaceAccess = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = surfaceAccess->GetInterface(winrt::guid_of<ID3D11Texture2D>(), reinterpret_cast<void**>(texture.GetAddressOf()));
    if (FAILED(hr) || !texture) {
        return false;
    }

    m_D3D11CapturedTexture = texture;

    // Zero-copy: Copy texture contents on D3D11 Context to our shared resource.
    // Wrap the DX12 shared resource as a D3D11 texture on the D3D11On12 Device.
    Microsoft::WRL::ComPtr<ID3D11On12Device> d3d11On12Device;
    m_D3D11Device.As(&d3d11On12Device);

    D3D11_RESOURCE_FLAGS flags11 = {};
    flags11.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    Microsoft::WRL::ComPtr<ID3D11Resource> wrapped11Resource;
    hr = d3d11On12Device->CreateWrappedResource(
        m_D3D12SharedResource.Get(),
        &flags11,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COMMON,
        IID_PPV_ARGS(&wrapped11Resource)
    );
    if (FAILED(hr)) return false;

    // Acquire and copy
    d3d11On12Device->AcquireWrappedResources(wrapped11Resource.GetAddressOf(), 1);
    m_D3D11Context->CopyResource(wrapped11Resource.Get(), m_D3D11CapturedTexture.Get());
    d3d11On12Device->ReleaseWrappedResources(wrapped11Resource.GetAddressOf(), 1);
    
    // Flush the context to commit changes
    m_D3D11Context->Flush();
    return true;
}

void WgcCapture::ReleaseFrame() {
    m_CurrentFrame = nullptr;
    m_D3D11CapturedTexture = nullptr;
}
