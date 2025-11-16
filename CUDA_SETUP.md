# CUDA Setup Guide for NeuroGen NLP Agent

## Overview

The NeuroGen NLP agent has been updated to support GPU acceleration via CUDA. This document explains how to enable CUDA support for significantly improved performance during training and inference.

## Current Build Configuration

The project is currently configured to build **WITHOUT** CUDA support, allowing it to compile on systems where CUDA is not installed. However, the code is ready to use CUDA when available.

## Performance Benefits

When CUDA is properly configured and available:
- **Neural network updates**: Run on GPU with parallel processing
- **Matrix operations**: Accelerated using cuBLAS
- **Token generation**: Faster logit computation and sampling
- **Training**: Significantly reduced iteration time

## Enabling CUDA Support

### Prerequisites

1. **CUDA Toolkit**: Install NVIDIA CUDA Toolkit (version 11.0 or later recommended)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Ensure `nvcc` is in your PATH
   - Verify installation: `nvcc --version`

2. **NVIDIA GPU**: Compatible NVIDIA GPU with compute capability 7.5 or higher
   - Check your GPU: `nvidia-smi`

3. **CUDA-compatible driver**: Latest NVIDIA drivers installed

### Build Configuration Steps

1. **Update Makefile** (`/home/user/NeuroGen-1.0/Makefile`):

   **Uncomment CUDA linking flags** (around line 30-37):
   ```makefile
   # Change from:
   # LDFLAGS := -L$(CUDA_PATH)/lib64 -L/usr/lib
   # LDLIBS := -ljsoncpp -lcudart -lcurand -lcublas -lcufft -lX11 -lXtst

   LDFLAGS := -L/usr/lib
   LDLIBS := -lX11

   # To:
   LDFLAGS := -L$(CUDA_PATH)/lib64 -L/usr/lib
   LDLIBS := -ljsoncpp -lcudart -lcurand -lcublas -lcufft -lX11 -lXtst

   # LDFLAGS := -L/usr/lib
   # LDLIBS := -lX11
   ```

   **Uncomment CUDA source files** (around line 65-82):
   ```makefile
   # Uncomment these lines:
   CUDA_WRAPPER_SOURCES := \
       $(CUDA_SRC_DIR)/CudaKernelWrappers.cu \
       $(CUDA_SRC_DIR)/EnhancedSTDPKernel.cu \
       $(CUDA_SRC_DIR)/HebbianLearningKernel.cu \
       $(CUDA_SRC_DIR)/HomeostaticMechanismsKernel.cu \
       $(CUDA_SRC_DIR)/NeuromodulationKernels.cu \
       $(CUDA_SRC_DIR)/LearningStateKernels.cu \
       $(CUDA_SRC_DIR)/KernelLaunchWrappers.cu \
       $(CUDA_SRC_DIR)/NetworkCUDA.cu \
       $(CUDA_SRC_DIR)/NetworkCUDA_Interface.cu

   CUDA_WRAPPER_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_WRAPPER_SOURCES))
   OBJECTS += $(CUDA_WRAPPER_OBJECTS)
   ```

2. **Add USE_CUDA compiler flag**:

   In the Makefile, update `CXXFLAGS` (around line 25):
   ```makefile
   CXXFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -O3 -g -fPIC -Wall -ferror-limit=50 -DUSE_CUDA
   ```

3. **Update excluded sources** (around line 40-45):

   Remove these from `EXCLUDE_SOURCES`:
   ```makefile
   # Remove or comment out:
   # $(SRC_DIR)/EnhancedLearningSystem.cpp \
   # $(SRC_DIR)/NetworkCUDA.cpp
   ```

4. **Rebuild**:
   ```bash
   make clean
   make -j4
   ```

## Verification

After building with CUDA enabled, run the NLP agent. You should see messages indicating CUDA initialization:

```
Initializing CUDA for module 'prefrontal_cortex' (found 1 CUDA device(s))
  Using GPU: NVIDIA GeForce RTX 3080 (Compute Capability: 8.6)
✅ CUDA initialization successful for module 'prefrontal_cortex'
  Neurons: 2048, Synapses: 81920
```

If CUDA is not available, you'll see:
```
ℹ️  Module 'prefrontal_cortex': CUDA support not enabled at compile-time. Using CPU processing.
  To enable CUDA: recompile with -DUSE_CUDA when CUDA toolkit is installed
```

## Architecture

### CUDA Integration Points

1. **NeuralModule** (`src/NeuralModule.cpp`):
   - Checks CUDA availability at initialization
   - Creates `NetworkCUDA_Interface` when CUDA is available
   - Falls back to CPU-based `Network` when CUDA is not available

2. **Network Processing**:
   - CPU path: Uses `Network` class for neural computations
   - GPU path: Uses `NetworkCUDA_Interface` → `NetworkCUDA` for GPU-accelerated processing

3. **Conditional Compilation**:
   - Code is wrapped in `#ifdef USE_CUDA` / `#endif` blocks
   - Compiles cleanly without CUDA
   - Automatically uses CUDA when available at runtime

### Key Files Modified

- `src/NeuralModule.cpp`: CUDA initialization and processing
- `include/NeuroGen/NeuralModule.h`: Conditional CUDA network member
- `Makefile`: Build configuration for CUDA

## Troubleshooting

### Build Errors

**Error**: `cuda_runtime.h not found`
- **Solution**: Install CUDA Toolkit or verify `CUDA_PATH` in Makefile points to correct location

**Error**: `cannot find -lcudart`
- **Solution**: Ensure CUDA libraries are in linker path. Update `LDFLAGS` in Makefile

### Runtime Errors

**Error**: `CUDA driver version is insufficient`
- **Solution**: Update NVIDIA drivers to match CUDA Toolkit version

**Error**: No CUDA devices found
- **Solution**: Verify GPU is properly installed and drivers are loaded (`nvidia-smi`)

## Performance Tips

1. **Batch Processing**: Process multiple inputs together for better GPU utilization
2. **Memory Management**: Monitor GPU memory usage during training
3. **Stream Optimization**: NetworkCUDA uses multiple CUDA streams for parallel processing
4. **Mixed Precision**: Can be enabled in `NetworkCUDA::CUDAConfig` for faster training

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA Deep Learning Best Practices](https://docs.nvidia.com/deeplearning/performance/)

## Contact

For issues specific to CUDA integration, check the project's issue tracker or review the implementation in `src/cuda/NetworkCUDA.cu`.
