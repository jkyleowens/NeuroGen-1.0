#include <NeuroGen/cuda/NetworkCUDA_Interface.h>

// --- Constructor ---
// Initializes the underlying NetworkCUDA object and copies initial data to the GPU.
NetworkCUDA_Interface::NetworkCUDA_Interface(
    const NetworkConfig& config,
    const std::vector<GPUNeuronState>& neurons,
    const std::vector<GPUSynapse>& synapses)
{
    // Create the CUDA network manager with default CUDA config.
    NetworkCUDA::CUDAConfig cuda_config;
    cuda_network_ = std::make_unique<NetworkCUDA>(cuda_config);
    
    // Initialize the CUDA network with the provided configuration.
    cuda_network_->initialize(config);

    // Note: NetworkCUDA doesn't have copy_to_gpu method.
    // The network structure is set up during initialization.
}

// --- Destructor ---
// The default destructor is sufficient as the unique_ptr will handle cleanup.
NetworkCUDA_Interface::~NetworkCUDA_Interface() = default;

// --- Simulation Step ---
// Delegates the step call to the underlying CUDA network object.
void NetworkCUDA_Interface::step(float current_time, float dt, float reward, const std::vector<float>& inputs) {
    if (cuda_network_) {
        // Use the update method instead of simulate_step
        cuda_network_->update(dt, reward);
        
        // Process inputs if provided
        if (!inputs.empty()) {
            cuda_network_->processInput(inputs);
        }
    }
}

// --- Get Statistics ---
// Fetches the latest stats from the CUDA network object.
NetworkStats NetworkCUDA_Interface::get_stats() const {
    NetworkStats stats = {};
    if (cuda_network_) {
        // NetworkCUDA doesn't have get_stats method that takes a reference
        // We'll need to implement this differently or return empty stats
        // For now, return empty stats structure
    }
    return stats;
}

// --- Get Full Network State ---
// Copies the entire state of neurons and synapses from the GPU to host memory.
void NetworkCUDA_Interface::get_network_state(std::vector<GPUNeuronState>& neurons, std::vector<GPUSynapse>& synapses) {
    if (cuda_network_) {
        // NetworkCUDA doesn't have copy_from_gpu method
        // We can get neuron states and synaptic weights separately
        auto neuron_states = cuda_network_->getNeuronStates();
        auto synaptic_weights = cuda_network_->getSynapticWeights();
        
        // Convert to GPUNeuronState and GPUSynapse formats if needed
        // For now, we'll leave this as a placeholder
        // TODO: Implement proper state extraction
    }
}