#include "NeuroGen/cuda/NetworkCUDA_Interface.h"
#include <iostream>
#include <stdexcept>

// Stub implementation for non-CUDA builds
// This provides the interface methods but with minimal functionality

NetworkCUDA_Interface::NetworkCUDA_Interface(
    const NetworkConfig& config,
    const std::vector<GPUNeuronState>& neurons,
    const std::vector<GPUSynapse>& synapses
) {
    std::cerr << "⚠️  Warning: CUDA support not compiled. Using stub implementation." << std::endl;
    std::cerr << "   Network will have limited functionality." << std::endl;
    std::cerr << "   To enable CUDA, compile with CUDA support." << std::endl;
    
    // Initialize with nullptr since we don't have CUDA
    cuda_network_ = nullptr;
}

NetworkCUDA_Interface::~NetworkCUDA_Interface() {
    // Cleanup handled by unique_ptr
}

void NetworkCUDA_Interface::step(
    float current_time,
    float dt,
    float reward,
    const std::vector<float>& inputs
) {
    // Stub implementation - does nothing
    // In a real CUDA build, this would call cuda_network_->update()
}

NetworkStats NetworkCUDA_Interface::get_stats() const {
    // Return empty/default stats
    NetworkStats stats;
    stats.total_neurons = 0;
    stats.total_synapses = 0;
    stats.total_spike_count = 0;
    stats.mean_firing_rate = 0.0f;
    stats.mean_synaptic_weight = 0.0f;
    stats.total_simulation_time = 0.0f;
    return stats;
}

void NetworkCUDA_Interface::get_network_state(
    std::vector<GPUNeuronState>& neurons,
    std::vector<GPUSynapse>& synapses
) {
    // Clear the vectors since we have no state to return
    neurons.clear();
    synapses.clear();
}
