#define NOMINMAX
#include "vulkan_swapchain.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>

// ─── Win32 Overlay Window ────────────────────────────────────────────────────

static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wp,
                                       LPARAM lp) {
  if (msg == WM_DESTROY) {
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

// Dedicated thread parameters for safe cross-thread window creation
static std::atomic<bool> g_OverlayRunning{false};
static std::atomic<HWND> g_OverlayHwndForThread{nullptr};
static std::atomic<bool> g_CreationDone{false};
static std::atomic<bool> g_CreationSuccess{false};
static std::thread g_MsgThread;

struct OverlayThreadParams {
  HWND targetHwnd;
  int displayWidth;
  int displayHeight;
  HINSTANCE hInst;
};

static void OverlayMessageThread(OverlayThreadParams params) {
  WNDCLASSEXW wc = {};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = OverlayWndProc;
  wc.hInstance = params.hInst;
  wc.lpszClassName = L"UpscalerrOverlay";
  RegisterClassExW(&wc); // ignore if already registered

  // Find the monitor where the game window is located
  HMONITOR hMonitor =
      MonitorFromWindow(params.targetHwnd, MONITOR_DEFAULTTONEAREST);
  MONITORINFO mi = {sizeof(mi)};
  GetMonitorInfoW(hMonitor, &mi);

  int monitorX = mi.rcMonitor.left;
  int monitorY = mi.rcMonitor.top;
  int monitorW = mi.rcMonitor.right - mi.rcMonitor.left;
  int monitorH = mi.rcMonitor.bottom - mi.rcMonitor.top;

  // Transparent overlay: clicks pass through to the game window underneath.
  // WS_EX_TRANSPARENT + WS_EX_LAYERED is required for proper click-through
  // across different processes on Windows 10/11.
  DWORD style = WS_POPUP | WS_VISIBLE;
  DWORD exStyle =
      WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_LAYERED;

  HWND hwnd = CreateWindowExW(
      exStyle, L"UpscalerrOverlay", L"Upscalerr Overlay", style, monitorX,
      monitorY, monitorW, monitorH, nullptr, nullptr, params.hInst, nullptr);

  if (!hwnd) {
    g_CreationSuccess = false;
    g_CreationDone = true;
    return;
  }

  ShowWindow(hwnd, SW_SHOW);
  UpdateWindow(hwnd);

  g_OverlayHwndForThread = hwnd;
  g_CreationSuccess = true;
  g_CreationDone = true;

  MSG msg;
  while (g_OverlayRunning.load()) {
    if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        break;
      }
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    Sleep(1);
  }

  DestroyWindow(hwnd);
  g_OverlayHwndForThread = nullptr;
}

bool VulkanSwapchain::CreateOverlayWindow(HWND targetHwnd, int displayWidth,
                                          int displayHeight) {
  (void)displayWidth;
  (void)displayHeight;
  m_HInst = GetModuleHandle(nullptr);

  g_OverlayRunning = true;
  g_CreationDone = false;
  g_CreationSuccess = false;
  g_OverlayHwndForThread = nullptr;

  OverlayThreadParams params = {targetHwnd, 0, 0, m_HInst};
  g_MsgThread = std::thread(OverlayMessageThread, params);

  // Wait for the window creation to finish on the thread
  while (!g_CreationDone.load()) {
    Sleep(5);
  }

  if (!g_CreationSuccess.load()) {
    if (g_MsgThread.joinable())
      g_MsgThread.join();
    return false;
  }

  m_OverlayHwnd = g_OverlayHwndForThread.load();

  std::cout << "[Vulkan] Fullscreen overlay window created on dedicated thread."
            << std::endl;
  return true;
}

void VulkanSwapchain::RepositionOverlay(HWND targetHwnd) {
  if (!m_OverlayHwnd || !targetHwnd)
    return;
  RECT rect = {};
  GetWindowRect(targetHwnd, &rect);
  SetWindowPos(m_OverlayHwnd, HWND_TOPMOST, rect.left, rect.top, m_OutWidth,
               m_OutHeight, SWP_NOACTIVATE);
}

// ─── Vulkan Instance ─────────────────────────────────────────────────────────

bool VulkanSwapchain::CreateInstance() {
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Upscalerr";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  const char *extensions[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
  };

  VkInstanceCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &appInfo;
  ci.enabledExtensionCount = 4;
  ci.ppEnabledExtensionNames = extensions;

  return vkCreateInstance(&ci, nullptr, &m_Instance) == VK_SUCCESS;
}

// ─── Device & Queue ──────────────────────────────────────────────────────────

bool VulkanSwapchain::CreateDeviceAndQueue() {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(m_Instance, &count, nullptr);
  if (count == 0)
    return false;

  std::vector<VkPhysicalDevice> devs(count);
  vkEnumeratePhysicalDevices(m_Instance, &count, devs.data());

  // Prefer discrete NVIDIA GPU
  for (auto &d : devs) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        strstr(p.deviceName, "NVIDIA")) {
      m_PhysDevice = d;
      break;
    }
  }
  if (!m_PhysDevice)
    m_PhysDevice = devs[0];

  // Find graphics queue
  uint32_t qCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_PhysDevice, &qCount, nullptr);
  std::vector<VkQueueFamilyProperties> qProps(qCount);
  vkGetPhysicalDeviceQueueFamilyProperties(m_PhysDevice, &qCount,
                                           qProps.data());

  bool found = false;
  for (uint32_t i = 0; i < qCount; ++i) {
    if (qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      m_QueueFamily = i;
      found = true;
      break;
    }
  }
  if (!found)
    return false;

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = m_QueueFamily;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  const char *devExts[] = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
  };

  VkDeviceCreateInfo dci = {};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 3;
  dci.ppEnabledExtensionNames = devExts;

  if (vkCreateDevice(m_PhysDevice, &dci, nullptr, &m_Device) != VK_SUCCESS)
    return false;
  vkGetDeviceQueue(m_Device, m_QueueFamily, 0, &m_Queue);
  return true;
}

// ─── Surface ─────────────────────────────────────────────────────────────────

bool VulkanSwapchain::CreateSurface() {
  // Surface is on the OVERLAY window — not the game window
  VkWin32SurfaceCreateInfoKHR si = {};
  si.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  si.hwnd = m_OverlayHwnd;
  si.hinstance = m_HInst;
  return vkCreateWin32SurfaceKHR(m_Instance, &si, nullptr, &m_Surface) ==
         VK_SUCCESS;
}

// ─── Swapchain ───────────────────────────────────────────────────────────────

bool VulkanSwapchain::CreateSwapchain() {
  // Verify surface capabilities
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_PhysDevice, m_Surface, &caps);

  // Choose FIFO (guaranteed) as fallback; prefer MAILBOX for lowest latency
  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  uint32_t pmCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_PhysDevice, m_Surface, &pmCount,
                                            nullptr);
  std::vector<VkPresentModeKHR> modes(pmCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_PhysDevice, m_Surface, &pmCount,
                                            modes.data());
  for (auto m : modes) {
    if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
      presentMode = m;
      break;
    }
  }

  // Swapchain is sized to match the actual display bounds of the overlay window
  RECT rect = {};
  GetWindowRect(m_OverlayHwnd, &rect);
  uint32_t width = rect.right - rect.left;
  uint32_t height = rect.bottom - rect.top;

  VkExtent2D extent = {width, height};
  extent.width =
      (std::max)(caps.minImageExtent.width,
                 (std::min)(caps.maxImageExtent.width, extent.width));
  extent.height =
      (std::max)(caps.minImageExtent.height,
                 (std::min)(caps.maxImageExtent.height, extent.height));

  uint32_t imageCount = (std::max)(caps.minImageCount, MAX_FRAMES_IN_FLIGHT);
  if (caps.maxImageCount > 0)
    imageCount = (std::min)(imageCount, caps.maxImageCount);

  VkSwapchainCreateInfoKHR sci = {};
  sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  sci.surface = m_Surface;
  sci.minImageCount = imageCount;
  sci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  sci.imageExtent = extent;
  sci.imageArrayLayers = 1;
  sci.imageUsage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sci.preTransform = caps.currentTransform;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sci.presentMode = presentMode;
  sci.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(m_Device, &sci, nullptr, &m_Swapchain) != VK_SUCCESS)
    return false;

  vkGetSwapchainImagesKHR(m_Device, m_Swapchain, &m_ImageCount, nullptr);
  m_SwapImages.resize(m_ImageCount);
  vkGetSwapchainImagesKHR(m_Device, m_Swapchain, &m_ImageCount,
                          m_SwapImages.data());

  return true;
}

// ─── Command Pool & Buffers ──────────────────────────────────────────────────

bool VulkanSwapchain::CreateCommandPoolAndBuffers() {
  VkCommandPoolCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.queueFamilyIndex = m_QueueFamily;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(m_Device, &ci, nullptr, &m_CmdPool) != VK_SUCCESS)
    return false;

  m_CmdBuffers.resize(m_ImageCount);
  VkCommandBufferAllocateInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = m_CmdPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = m_ImageCount;

  return vkAllocateCommandBuffers(m_Device, &ai, m_CmdBuffers.data()) ==
         VK_SUCCESS;
}

// ─── Sync Objects ────────────────────────────────────────────────────────────

bool VulkanSwapchain::CreateSyncObjects() {
  m_ImageAvailable.resize(MAX_FRAMES_IN_FLIGHT);
  m_RenderFinished.resize(MAX_FRAMES_IN_FLIGHT);
  m_InFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

  VkSemaphoreCreateInfo semi = {};
  semi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fci = {};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first frame
                                            // doesn't block

  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    if (vkCreateSemaphore(m_Device, &semi, nullptr, &m_ImageAvailable[i]) !=
        VK_SUCCESS)
      return false;
    if (vkCreateSemaphore(m_Device, &semi, nullptr, &m_RenderFinished[i]) !=
        VK_SUCCESS)
      return false;
    if (vkCreateFence(m_Device, &fci, nullptr, &m_InFlightFences[i]) !=
        VK_SUCCESS)
      return false;
  }
  return true;
}

// ─── Shared Vulkan Image (CUDA writes here)
// ───────────────────────────────────

bool VulkanSwapchain::CreateSharedVulkanImage() {
  VkExternalMemoryImageCreateInfo extImgInfo = {};
  extImgInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  extImgInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  VkImageCreateInfo ici = {};
  ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ici.pNext = &extImgInfo;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_B8G8R8A8_UNORM;
  ici.extent = {static_cast<uint32_t>(m_OutWidth),
                static_cast<uint32_t>(m_OutHeight), 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (vkCreateImage(m_Device, &ici, nullptr, &m_SharedImage) != VK_SUCCESS)
    return false;

  VkMemoryRequirements memReq;
  vkGetImageMemoryRequirements(m_Device, m_SharedImage, &memReq);

  VkExportMemoryAllocateInfo exportInfo = {};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(m_PhysDevice, &memProps);

  uint32_t memTypeIdx = UINT32_MAX;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((memReq.memoryTypeBits & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      memTypeIdx = i;
      break;
    }
  }
  if (memTypeIdx == UINT32_MAX)
    return false;

  VkMemoryAllocateInfo mai = {};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.pNext = &exportInfo;
  mai.allocationSize = memReq.size;
  mai.memoryTypeIndex = memTypeIdx;

  if (vkAllocateMemory(m_Device, &mai, nullptr, &m_SharedMemory) != VK_SUCCESS)
    return false;
  vkBindImageMemory(m_Device, m_SharedImage, m_SharedMemory, 0);

  // Get Win32 exportable handle
  auto fpGetHandle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
      m_Device, "vkGetMemoryWin32HandleKHR");
  if (!fpGetHandle)
    return false;

  VkMemoryGetWin32HandleInfoKHR handleInfo = {};
  handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  handleInfo.memory = m_SharedMemory;
  handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  if (fpGetHandle(m_Device, &handleInfo, &m_SharedHandle) != VK_SUCCESS)
    return false;
  m_SharedSize = memReq.size;

  // Transition the shared image to GENERAL layout so CUDA can write to it
  // immediately
  VkCommandBufferAllocateInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = m_CmdPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;

  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(m_Device, &ai, &cmd) == VK_SUCCESS) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    TransitionImage(cmd, m_SharedImage, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    VK_ACCESS_TRANSFER_WRITE_BIT);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vkQueueSubmit(m_Queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_Queue);
    vkFreeCommandBuffers(m_Device, m_CmdPool, 1, &cmd);
  }

  return true;
}

// ─── CUDA Import ─────────────────────────────────────────────────────────────

bool VulkanSwapchain::MapVulkanImageToCuda() {
  cudaExternalMemoryHandleDesc extDesc = {};
  extDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  extDesc.handle.win32.handle = m_SharedHandle;
  extDesc.size = m_SharedSize;

  if (cudaImportExternalMemory(&m_CudaExtMem, &extDesc) != cudaSuccess)
    return false;

  cudaExternalMemoryMipmappedArrayDesc mipDesc = {};
  mipDesc.offset = 0;
  mipDesc.formatDesc.x = 8;
  mipDesc.formatDesc.y = 8;
  mipDesc.formatDesc.z = 8;
  mipDesc.formatDesc.w = 8;
  mipDesc.formatDesc.f = cudaChannelFormatKindUnsigned;
  mipDesc.extent.width = static_cast<size_t>(m_OutWidth);
  mipDesc.extent.height = static_cast<size_t>(m_OutHeight);
  mipDesc.extent.depth = 0;
  mipDesc.numLevels = 1;
  mipDesc.flags =
      cudaArraySurfaceLoadStore; // CRITICAL: enables surface read/write

  if (cudaExternalMemoryGetMappedMipmappedArray(
          &m_CudaMipmappedArray, m_CudaExtMem, &mipDesc) != cudaSuccess)
    return false;
  if (cudaGetMipmappedArrayLevel(&m_CudaArray, m_CudaMipmappedArray, 0) !=
      cudaSuccess)
    return false;

  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_CudaArray;

  return cudaCreateSurfaceObject(&m_CudaSurface, &resDesc) == cudaSuccess;
}

// ─── Image Layout Transition Helper ──────────────────────────────────────────

void VulkanSwapchain::TransitionImage(VkCommandBuffer cmd, VkImage image,
                                      VkImageLayout from, VkImageLayout to,
                                      VkPipelineStageFlags srcStage,
                                      VkPipelineStageFlags dstStage,
                                      VkAccessFlags srcAccess,
                                      VkAccessFlags dstAccess) {
  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = from;
  barrier.newLayout = to;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = srcAccess;
  barrier.dstAccessMask = dstAccess;

  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
}

// ─── Initialize ──────────────────────────────────────────────────────────────

VulkanSwapchain::VulkanSwapchain() {}
VulkanSwapchain::~VulkanSwapchain() { Cleanup(); }

bool VulkanSwapchain::Initialize(HWND targetHwnd, int captureWidth,
                                 int captureHeight, int scaleFactor) {
  Cleanup();

  m_CaptureWidth = captureWidth;
  m_CaptureHeight = captureHeight;
  m_ScaleFactor = scaleFactor;
  m_OutWidth = captureWidth * scaleFactor;
  m_OutHeight = captureHeight * scaleFactor;

  // Create a fullscreen borderless window on the monitor containing targetHwnd.
  if (!CreateOverlayWindow(targetHwnd, 0, 0)) {
    std::cerr << "[Vulkan] Failed to create overlay window." << std::endl;
    return false;
  }

  // Step 2: Vulkan setup against the overlay window
  if (!CreateInstance()) {
    std::cerr << "[Vulkan] CreateInstance failed.\n";
    return false;
  }
  if (!CreateDeviceAndQueue()) {
    std::cerr << "[Vulkan] CreateDevice failed.\n";
    return false;
  }
  if (!CreateSurface()) {
    std::cerr << "[Vulkan] CreateSurface failed.\n";
    return false;
  }
  if (!CreateSwapchain()) {
    std::cerr << "[Vulkan] CreateSwapchain failed.\n";
    return false;
  }
  if (!CreateCommandPoolAndBuffers()) {
    std::cerr << "[Vulkan] CreateCmdPool failed.\n";
    return false;
  }
  if (!CreateSyncObjects()) {
    std::cerr << "[Vulkan] CreateSync failed.\n";
    return false;
  }

  // Step 3: Shared image that CUDA writes upscaled frames into
  if (!CreateSharedVulkanImage()) {
    std::cerr << "[Vulkan] SharedImage failed.\n";
    return false;
  }
  if (!MapVulkanImageToCuda()) {
    std::cerr << "[Vulkan] CUDA import failed.\n";
    return false;
  }

  std::cout << "[Vulkan] Borderless presentation pipeline registered."
            << std::endl;
  return true;
}

// ─── Present ─────────────────────────────────────────────────────────────────

void VulkanSwapchain::Present() {
  if (!m_Device || !m_Swapchain)
    return;

  // Pump messages is now handled by the dedicated overlay thread;
  // no need to do it here.

  uint32_t frameIdx = m_FrameIndex % MAX_FRAMES_IN_FLIGHT;

  // Wait for the previous use of this frame slot to finish (100ms timeout to
  // avoid deadlock)
  VkResult waitRes = vkWaitForFences(m_Device, 1, &m_InFlightFences[frameIdx],
                                     VK_TRUE, 100'000'000ULL);
  if (waitRes == VK_TIMEOUT) {
    std::cerr << "[Vulkan] Present: fence timeout, skipping frame."
              << std::endl;
    return;
  }
  vkResetFences(m_Device, 1, &m_InFlightFences[frameIdx]);

  // Acquire next swapchain image — signal m_ImageAvailable when ready
  uint32_t imageIdx;
  VkResult res = vkAcquireNextImageKHR(m_Device, m_Swapchain, UINT64_MAX,
                                       m_ImageAvailable[frameIdx],
                                       VK_NULL_HANDLE, &imageIdx);
  if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
    return;
  if (res != VK_SUCCESS)
    return;

  // Record command buffer: shared image → swapchain image
  VkCommandBuffer cmd = m_CmdBuffers[imageIdx];
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  // Transition shared image: GENERAL → TRANSFER_SRC (CUDA has finished writing
  // to GENERAL layout)
  TransitionImage(cmd, m_SharedImage, VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT);

  // Transition swapchain image: UNDEFINED → TRANSFER_DST
  TransitionImage(
      cmd, m_SwapImages[imageIdx], VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);

  // Blit: full CUDA output (OutWidth×OutHeight) → Fullscreen Swapchain overlay
  // Fetch current overlay size to construct correct blit offset
  RECT rect = {};
  GetWindowRect(m_OverlayHwnd, &rect);
  int monitorWidth = rect.right - rect.left;
  int monitorHeight = rect.bottom - rect.top;

  VkImageBlit blit = {};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[0] = {0, 0, 0};
  blit.srcOffsets[1] = {m_OutWidth, m_OutHeight, 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[0] = {0, 0, 0};
  blit.dstOffsets[1] = {monitorWidth, monitorHeight, 1};

  vkCmdBlitImage(cmd, m_SharedImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 m_SwapImages[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 1, &blit, VK_FILTER_LINEAR);

  // Transition swapchain image → PRESENT_SRC
  TransitionImage(
      cmd, m_SwapImages[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0);

  // Transition shared image back to GENERAL layout so CUDA can write to it next
  // frame
  TransitionImage(cmd, m_SharedImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT);

  vkEndCommandBuffer(cmd);

  // Submit: wait for imageAvailable, signal renderFinished
  VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo si = {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &m_ImageAvailable[frameIdx];
  si.pWaitDstStageMask = &waitStage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &m_RenderFinished[frameIdx];

  vkQueueSubmit(m_Queue, 1, &si, m_InFlightFences[frameIdx]);

  // Present
  VkPresentInfoKHR pi = {};
  pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &m_RenderFinished[frameIdx];
  pi.swapchainCount = 1;
  pi.pSwapchains = &m_Swapchain;
  pi.pImageIndices = &imageIdx;

  vkQueuePresentKHR(m_Queue, &pi);

  m_FrameIndex++;
}

// ─── Cleanup ─────────────────────────────────────────────────────────────────

void VulkanSwapchain::Cleanup() {
  if (m_Device)
    vkDeviceWaitIdle(m_Device);

  // CUDA resources
  if (m_CudaSurface) {
    cudaDestroySurfaceObject(m_CudaSurface);
    m_CudaSurface = 0;
  }
  if (m_CudaMipmappedArray) {
    cudaFreeMipmappedArray(m_CudaMipmappedArray);
    m_CudaMipmappedArray = nullptr;
  }
  if (m_CudaExtMem) {
    cudaDestroyExternalMemory(m_CudaExtMem);
    m_CudaExtMem = nullptr;
  }

  // Win32 handle
  if (m_SharedHandle) {
    CloseHandle(m_SharedHandle);
    m_SharedHandle = nullptr;
  }

  // Vulkan shared image
  if (m_SharedImage) {
    vkDestroyImage(m_Device, m_SharedImage, nullptr);
    m_SharedImage = VK_NULL_HANDLE;
  }
  if (m_SharedMemory) {
    vkFreeMemory(m_Device, m_SharedMemory, nullptr);
    m_SharedMemory = VK_NULL_HANDLE;
  }

  // Sync objects
  for (uint32_t i = 0; i < m_ImageAvailable.size(); ++i) {
    vkDestroySemaphore(m_Device, m_ImageAvailable[i], nullptr);
    vkDestroySemaphore(m_Device, m_RenderFinished[i], nullptr);
    vkDestroyFence(m_Device, m_InFlightFences[i], nullptr);
  }
  m_ImageAvailable.clear();
  m_RenderFinished.clear();
  m_InFlightFences.clear();

  // Command pool (frees all command buffers automatically)
  if (m_CmdPool) {
    vkDestroyCommandPool(m_Device, m_CmdPool, nullptr);
    m_CmdPool = VK_NULL_HANDLE;
  }
  m_CmdBuffers.clear();
  m_SwapImages.clear();

  if (m_Swapchain) {
    vkDestroySwapchainKHR(m_Device, m_Swapchain, nullptr);
    m_Swapchain = VK_NULL_HANDLE;
  }
  if (m_Surface) {
    vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
    m_Surface = VK_NULL_HANDLE;
  }
  if (m_Device) {
    vkDestroyDevice(m_Device, nullptr);
    m_Device = VK_NULL_HANDLE;
  }
  if (m_Instance) {
    vkDestroyInstance(m_Instance, nullptr);
    m_Instance = VK_NULL_HANDLE;
  }

  // Overlay window
  if (m_OverlayHwnd) {
    // Stop the dedicated message thread first
    g_OverlayRunning = false;
    if (g_MsgThread.joinable())
      g_MsgThread.join();

    DestroyWindow(m_OverlayHwnd);
    m_OverlayHwnd = nullptr;
  }
}
