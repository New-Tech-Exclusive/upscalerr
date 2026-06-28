import struct
import win32file
import win32pipe
import pywintypes

# IPC commands mapping to protocol.h
CMD_TOGGLE_UP = 0
CMD_TOGGLE_FG = 1
CMD_SET_SCALE = 2
CMD_SET_TARGET = 3
CMD_GET_STATS = 4
CMD_SHUTDOWN = 5

class IpcClient:
    def __init__(self, pipe_name=r"\\.\pipe\upscalerr"):
        self.pipe_name = pipe_name

    def _send_command(self, cmd_type: int, payload_val: int) -> tuple[int, bytes]:
        """
        Packs IpcCommand struct and sends it over the named pipe, returning response status and bytes.
        Structure:
          IpcCommand:
            - type: uint32 (IpcCommandType)
            - payload: union/uint64 (8 bytes payload)
          Total: 12 bytes
        """
        # Pack format: '<IQ' -> little-endian, no padding
        # I: uint32 (4 bytes), Q: uint64 (8 bytes) = 12 bytes total
        # Must use '<' prefix to match C++ #pragma pack(push, 1) layout
        command_bytes = struct.pack("<IQ", cmd_type, payload_val)
        
        try:
            # Connect to Named Pipe
            handle = win32file.CreateFile(
                self.pipe_name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0, None,
                win32file.OPEN_EXISTING,
                0, None
            )
            
            # Set read mode to MESSAGE
            win32pipe.SetNamedPipeHandleState(
                handle,
                win32pipe.PIPE_READMODE_MESSAGE,
                None, None
            )

            # Write the command
            win32file.WriteFile(handle, command_bytes)
            
            # Read the response
            # Response:
            #   - status: uint32 (4 bytes)
            #   - payload: union/struct (8 bytes)
            # Total: 12 bytes
            err, resp_bytes = win32file.ReadFile(handle, 12)
            win32file.CloseHandle(handle)
            
            if err == 0:
                resp_status = struct.unpack("<I", resp_bytes[:4])[0]
                return resp_status, resp_bytes[4:]
            
        except pywintypes.error as e:
            # Pipe connection/busy error
            pass
        
        return 1, b"" # Return RESP_ERROR on exception/failure

    def toggle_upscale(self, enabled: bool) -> bool:
        status, _ = self._send_command(CMD_TOGGLE_UP, 1 if enabled else 0)
        return status == 0

    def toggle_frame_gen(self, enabled: bool) -> bool:
        status, _ = self._send_command(CMD_TOGGLE_FG, 1 if enabled else 0)
        return status == 0

    def set_scale_factor(self, scale: int) -> bool:
        status, _ = self._send_command(CMD_SET_SCALE, scale)
        return status == 0

    def set_target_window(self, hwnd: int) -> bool:
        status, _ = self._send_command(CMD_SET_TARGET, hwnd)
        return status == 0

    def get_stats(self) -> tuple[float, float]:
        """
        Returns (FPS, Latency in ms) retrieved from C++ backend
        """
        status, data = self._send_command(CMD_GET_STATS, 0)
        if status == 0 and len(data) >= 8:
            # Unpack response payload structure: IpcStats (float, float) -> 'ff'
            fps, latency = struct.unpack("<ff", data[:8])
            return fps, latency
        return 0.0, 0.0

    def request_shutdown(self) -> bool:
        status, _ = self._send_command(CMD_SHUTDOWN, 0)
        return status == 0
