#include "pipe_server.h"
#include <iostream>

PipeServer::PipeServer(const std::wstring& pipeName, IpcCallback callback) {
    m_PipeName = pipeName;
    m_Callback = callback;
    m_Running = false;
}

PipeServer::~PipeServer() {
    Stop();
}

bool PipeServer::Start() {
    if (m_Running) return true;

    m_Running = true;
    m_ServerThread = std::thread(&PipeServer::ServerLoop, this);
    return true;
}

void PipeServer::Stop() {
    if (!m_Running) return;

    m_Running = false;

    // Trigger pipe shutdown by opening a client connection to release blocker
    HANDLE clientPipe = CreateFileW(
        m_PipeName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, nullptr,
        OPEN_EXISTING,
        0, nullptr
    );
    if (clientPipe != INVALID_HANDLE_VALUE) {
        CloseHandle(clientPipe);
    }

    if (m_ServerThread.joinable()) {
        m_ServerThread.join();
    }
}

void PipeServer::ServerLoop() {
    while (m_Running) {
        // Create Named Pipe instance
        m_PipeHandle = CreateNamedPipeW(
            m_PipeName.c_str(),
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            sizeof(IpcResponse),
            sizeof(IpcCommand),
            0, nullptr
        );

        if (m_PipeHandle == INVALID_HANDLE_VALUE) {
            std::cerr << "[IPC] CreateNamedPipeW failed: " << GetLastError() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        // Wait for a client to connect (blocking call)
        BOOL connected = ConnectNamedPipe(m_PipeHandle, nullptr) ? 
            TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);

        if (connected && m_Running) {
            IpcCommand command = {};
            DWORD bytesRead = 0;

            // Read the binary command payload
            BOOL readSuccess = ReadFile(
                m_PipeHandle,
                &command,
                sizeof(IpcCommand),
                &bytesRead,
                nullptr
            );

            if (readSuccess && bytesRead == sizeof(IpcCommand)) {
                IpcResponse response = {};
                
                // Call processing callback
                m_Callback(command, response);

                // Write response packet back to client
                DWORD bytesWritten = 0;
                WriteFile(
                    m_PipeHandle,
                    &response,
                    sizeof(IpcResponse),
                    &bytesWritten,
                    nullptr
                );
            }
        }

        // Flush and close instance
        DisconnectNamedPipe(m_PipeHandle);
        CloseHandle(m_PipeHandle);
        m_PipeHandle = INVALID_HANDLE_VALUE;
    }
}
