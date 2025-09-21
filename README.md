# CUDA Multiband Blender

This project demonstrates image stitching and blending using OpenCV with CUDA acceleration.  
The goal is to compare different blending methods to show tradeoffs in speed vs quality.
I will be implementing multiband blending from scratch, and comparing it with feathering or perhaps a simple gradient blend.  

I built this project as part of a technical interview demo.

### OpenCV Build Notes

This project uses a custom OpenCV build with CUDA support.  
Below are the exact versions and steps I used (Windows, RTX 3080):

**Dependencies downloaded:**
- OpenCV source code v4.9.0
- OpenCV contrib v4.9.0 (must match source version)
- CUDA Toolkit v12.3
- cuDNN v8.97
- CMake
- Visual Studio 2022 (with C++ options enabled)

**Setup:**
1. Copy cuDNN files into `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/`.
2. Create `C:/OpenCV_GPU/` with subfolders:
   - `build/`
   - `install/`
3. Move `opencv/` and `opencv_contrib/` into `C:/OpenCV_GPU/`.
4. Open CMake, set:
   - Source: `opencv-4.9.0/`
   - Destination: `build/`
   - Check “Grouped” option
   - Click *Configure*.

**Enable in CMake-GUI:**
- WITH_CUDA
- ENABLE_FAST_MATH
- BUILD_OPENCV_WORLD
- OPENCV_EXTRA_MODULES_PATH → `opencv_contrib/modules`
- OPENCV_DNN_CUDA
- BUILD_OPENCV_DNN

**Then enable:**
- CUDA_FAST_MATH
- CUDA_ARCH_BIN = 8.6 (RTX 3080)
- CMAKE_INSTALL_DIRECTORY → `install/`
- CMAKE_CONFIGURATION_TYPES = Release

**Generate & Build:**
```powershell
"C:\Programming Files\CMake\bin\cmake.exe" --build "../OpenCV_GPU/build" --target INSTALL --config Release