#pragma once

#include <stdint.h>

#pragma pack(push, 1)

// Command identifier definitions
enum IpcCommandType : uint32_t {
    CMD_TOGGLE_UP    = 0, // Toggle Upscaling (ON/OFF)
    CMD_TOGGLE_FG    = 1, // Toggle Frame Generation (ON/OFF)
    CMD_SET_SCALE    = 2, // Set Scale Factor (2x, 3x, 4x)
    CMD_SET_TARGET   = 3, // Set capture target Win32 HWND
    CMD_GET_STATS    = 4, // Retrieve performance stats (FPS, Latency)
    CMD_SHUTDOWN     = 5  // Request backend exit
};

// Target parameters passed in CMD structures
union IpcPayload {
    uint32_t toggleValue;     // Used by CMD_TOGGLE_UP, CMD_TOGGLE_FG (1=ON, 0=OFF)
    uint32_t scaleFactor;     // Used by CMD_SET_SCALE (2, 3, 4)
    uint64_t windowHandle;    // Used by CMD_SET_TARGET (HWND cast to uint64)
};

// Input command block
struct IpcCommand {
    IpcCommandType type;
    IpcPayload payload;
};

// Output status codes
enum IpcResponseStatus : uint32_t {
    RESP_OK    = 0,
    RESP_ERROR = 1
};

// Stat reporting telemetry values
struct IpcStats {
    float fps;
    float latency_ms;
};

union IpcResponsePayload {
    IpcStats stats;
};

// Response structure
struct IpcResponse {
    IpcResponseStatus status;
    IpcResponsePayload payload;
};

#pragma pack(pop)
