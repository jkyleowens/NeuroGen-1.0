#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include "NeuroGen/BrainModuleArchitecture.h"
#include <mutex>

void NetworkCUDA::setBrainArchitecture(std::shared_ptr<BrainModuleArchitecture> architecture) {
    // Fallback non-CUDA implementation (if full CUDA file also linked, this will be ODR-resolved by inline semantics)
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    brain_architecture_ = architecture;
}
