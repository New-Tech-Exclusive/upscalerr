#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <Windows.h>
#include "protocol.h"

typedef std::function<void(const IpcCommand&, IpcResponse&)> IpcCallback;

class PipeServer {
public:
    PipeServer(const std::wstring& pipeName, IpcCallback callback);
    ~PipeServer();

    bool Start();
    void Stop();

private:
    void ServerLoop();

    std::wstring m_PipeName;
    IpcCallback m_Callback;
    std::thread m_ServerThread;
    std::atomic<bool> m_Running;
    HANDLE m_PipeHandle = INVALID_HANDLE_VALUE;
};
