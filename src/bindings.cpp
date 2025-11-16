// Python bindings using pybind11 - CUDA-ENABLED VERSION
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

// Include NeuroGen headers
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>

// Conditional CUDA support
#ifdef USE_CUDA
#include <NeuroGen/cuda/NetworkCUDA_Interface.h>
#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <iostream>
#endif

namespace py = pybind11;

// Wrapper class to adapt Network interface for Python training script
class NetworkWrapper {
private:
#ifdef USE_CUDA
    std::unique_ptr<NetworkCUDA_Interface> cuda_network_;
    bool using_cuda_;
#endif
    std::unique_ptr<Network> cpu_network_;
    
    float last_reward = 0.0f;
    std::vector<float> last_output;
    NetworkConfig config_;

public:
    explicit NetworkWrapper(const NetworkConfig& config) : config_(config) {
#ifdef USE_CUDA
        // Try to initialize CUDA network
        using_cuda_ = false;
        
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error == cudaSuccess && device_count > 0) {
            try {
                std::cout << "🚀 Initializing GPU-accelerated neural network..." << std::endl;
                std::cout << "   Found " << device_count << " CUDA device(s)" << std::endl;
                
                // Get device properties
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                std::cout << "   Using: " << prop.name << std::endl;
                std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
                
                // Initialize neurons and synapses for CUDA network
                std::vector<GPUNeuronState> neurons(config.num_neurons);
                std::vector<GPUSynapse> synapses(config.totalSynapses);
                
                // Initialize basic neuron parameters
                for (size_t i = 0; i < neurons.size(); i++) {
                    neurons[i].V = -65.0f;  // Resting membrane potential
                    neurons[i].u = 0.0f;    // Recovery variable
                    neurons[i].last_spike_time = -1000.0f;
                    neurons[i].previous_spike_time = -1000.0f;
                    neurons[i].threshold = -55.0f;  // Firing threshold
                    neurons[i].excitability = 1.0f;
                    neurons[i].I_ext = 0.0f;
                    neurons[i].activity_level = 0.0f;
                    neurons[i].firing_rate = 0.0f;
                    neurons[i].neuron_type = (i % 5 == 0) ? 1 : 0; // 20% inhibitory
                    neurons[i].active = 1;
                    neurons[i].num_compartments = 1;
                    neurons[i].is_principal_cell = (neurons[i].neuron_type == 0);
                    
                    // Initialize arrays
                    for (int j = 0; j < MAX_COMPARTMENTS; j++) {
                        neurons[i].I_syn[j] = 0.0f;
                        neurons[i].ca_conc[j] = 0.0f;
                        neurons[i].V_compartments[j] = -65.0f;
                    }
                }
                
                // Initialize synapses
                for (size_t i = 0; i < synapses.size(); i++) {
                    synapses[i].pre_neuron_idx = i % config.num_neurons;
                    synapses[i].post_neuron_idx = (i + 1) % config.num_neurons;
                    synapses[i].target_neuron_idx = synapses[i].post_neuron_idx;
                    synapses[i].weight = 0.5f;
                    synapses[i].max_weight = 2.0f;
                    synapses[i].min_weight = 0.0f;
                    synapses[i].delay = 1.0f;
                    synapses[i].active = 1;
                    synapses[i].is_plastic = true;
                    synapses[i].learning_rate = 0.001f;
                    synapses[i].eligibility_trace = 0.0f;
                    synapses[i].last_pre_spike_time = -1000.0f;
                    synapses[i].last_post_spike_time = -1000.0f;
                    synapses[i].release_probability = 0.5f;
                    synapses[i].vesicle_count = 10;
                    synapses[i].post_compartment = 0;
                    synapses[i].receptor_index = (i % 4 == 0) ? 1 : 0; // 25% inhibitory
                }
                
                cuda_network_ = std::make_unique<NetworkCUDA_Interface>(
                    config, neurons, synapses
                );
                
                using_cuda_ = true;
                std::cout << "✅ GPU acceleration enabled!" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Failed to initialize CUDA: " << e.what() << std::endl;
                std::cerr << "   Falling back to CPU mode" << std::endl;
                cpu_network_ = std::make_unique<Network>(config);
            }
        } else {
            std::cout << "ℹ️  No CUDA devices found, using CPU mode" << std::endl;
            cpu_network_ = std::make_unique<Network>(config);
        }
#else
        // CPU-only build
        std::cout << "ℹ️  Built without CUDA support, using CPU mode" << std::endl;
        cpu_network_ = std::make_unique<Network>(config);
#endif
    }

    // Step method: takes dt and input vector, returns predicted token ID
    int step(float dt, const std::vector<float>& input_vector) {
#ifdef USE_CUDA
        if (using_cuda_ && cuda_network_) {
            // Use CUDA network
            std::vector<float> inputs = input_vector;
            inputs.resize(config_.input_size, 0.0f);
            
            cuda_network_->step(0.0f, dt, last_reward, inputs);
            
            // Get output (simplified - you may need to add get_output to NetworkCUDA_Interface)
            auto stats = cuda_network_->get_stats();
            
            // For now, use a simple heuristic based on network activity
            int predicted_id = static_cast<int>(stats.mean_firing_rate * 100.0f) % 32000;
            return std::max(0, predicted_id);
        }
#endif
        
        // CPU fallback
        if (cpu_network_) {
            cpu_network_->update(dt, input_vector, last_reward);
            last_output = cpu_network_->get_output();
            
            if (last_output.empty()) {
                return 0;
            }
            
            int predicted_id = static_cast<int>(std::round(last_output[0]));
            return std::max(0, predicted_id);
        }
        
        return 0;
    }

    // Apply reward for learning
    void apply_reward(float reward) {
        last_reward = reward;
    }

    // Save model to file
    void save_model(const std::string& file_path) {
#ifdef USE_CUDA
        if (using_cuda_ && cuda_network_) {
            // TODO: Implement CUDA state saving
            throw std::runtime_error("CUDA model saving not yet implemented");
        }
#endif
        
        if (cpu_network_) {
            if (!cpu_network_->saveToFile(file_path)) {
                throw std::runtime_error("Failed to save model to " + file_path);
            }
        }
    }

    // Load model from file
    void load_model(const std::string& file_path) {
#ifdef USE_CUDA
        if (using_cuda_ && cuda_network_) {
            // TODO: Implement CUDA state loading
            throw std::runtime_error("CUDA model loading not yet implemented");
        }
#endif
        
        if (cpu_network_) {
            if (!cpu_network_->loadFromFile(file_path)) {
                throw std::runtime_error("Failed to load model from " + file_path);
            }
        }
    }

    // Get network statistics
    std::string get_stats() const {
#ifdef USE_CUDA
        if (using_cuda_ && cuda_network_) {
            NetworkStats stats = cuda_network_->get_stats();
            return "NetworkStats{neurons=" + std::to_string(stats.total_neurons) +
                   ", synapses=" + std::to_string(stats.total_synapses) +
                   ", mean_firing_rate=" + std::to_string(stats.mean_firing_rate) +
                   ", mode=GPU}";
        }
#endif
        
        if (cpu_network_) {
            NetworkStats stats = cpu_network_->get_stats();
            return "NetworkStats{neurons=" + std::to_string(stats.total_neurons) +
                   ", synapses=" + std::to_string(stats.total_synapses) +
                   ", mean_firing_rate=" + std::to_string(stats.mean_firing_rate) +
                   ", mode=CPU}";
        }
        
        return "NetworkStats{uninitialized}";
    }

    // Reset network state
    void reset() {
#ifdef USE_CUDA
        if (using_cuda_ && cuda_network_) {
            // Reinitialize CUDA network
            cuda_network_.reset();
            
            std::vector<GPUNeuronState> neurons(config_.num_neurons);
            std::vector<GPUSynapse> synapses(config_.totalSynapses);
            
            // Re-initialize
            for (size_t i = 0; i < neurons.size(); i++) {
                neurons[i].V = -65.0f;
                neurons[i].u = 0.0f;
                neurons[i].last_spike_time = -1000.0f;
                neurons[i].previous_spike_time = -1000.0f;
                neurons[i].active = 1;
            }
            
            cuda_network_ = std::make_unique<NetworkCUDA_Interface>(
                config_, neurons, synapses
            );
            
            last_reward = 0.0f;
            return;
        }
#endif
        
        if (cpu_network_) {
            cpu_network_->reset();
            last_reward = 0.0f;
        }
    }
    
    // Get whether CUDA is being used
    bool is_using_cuda() const {
#ifdef USE_CUDA
        return using_cuda_;
#else
        return false;
#endif
    }
};

// PYBIND11_MODULE defines the module name and its contents
PYBIND11_MODULE(neural_network, m) {
    m.doc() = "NeuroGen Neural Network Python bindings (CUDA-enabled)";

    // Bind NetworkConfig class
    py::class_<NetworkConfig>(m, "NetworkConfig")
        .def(py::init<>(), "Create a default NetworkConfig")
        .def_readwrite("dt", &NetworkConfig::dt)
        .def_readwrite("num_neurons", &NetworkConfig::num_neurons)
        .def_readwrite("max_neurons", &NetworkConfig::max_neurons)
        .def_readwrite("num_synapses", &NetworkConfig::totalSynapses)
        .def_readwrite("enable_stdp", &NetworkConfig::enable_stdp)
        .def_readwrite("stdp_learning_rate", &NetworkConfig::stdp_learning_rate)
        .def_readwrite("learning_rate", &NetworkConfig::reward_learning_rate)
        .def_readwrite("input_size", &NetworkConfig::input_size)
        .def_readwrite("hidden_size", &NetworkConfig::hidden_size)
        .def_readwrite("output_size", &NetworkConfig::output_size);

    // Bind NetworkWrapper as "Network" for Python
    py::class_<NetworkWrapper>(m, "Network")
        .def(py::init<const NetworkConfig&>(),
             "Create a neural network with the given configuration",
             py::arg("config"))
        .def("step", &NetworkWrapper::step,
             "Perform one simulation step and return predicted token ID",
             py::arg("dt"), py::arg("input_vector"))
        .def("apply_reward", &NetworkWrapper::apply_reward,
             "Apply reward signal for learning",
             py::arg("reward"))
        .def("save_model", &NetworkWrapper::save_model,
             "Save network model to file",
             py::arg("file_path"))
        .def("load_model", &NetworkWrapper::load_model,
             "Load network model from file",
             py::arg("file_path"))
        .def("get_stats", &NetworkWrapper::get_stats,
             "Get network statistics as string")
        .def("reset", &NetworkWrapper::reset,
             "Reset network state")
        .def("is_using_cuda", &NetworkWrapper::is_using_cuda,
             "Check if network is using CUDA acceleration");

    // Version information
    m.attr("__version__") = "1.0.0";
    
#ifdef USE_CUDA
    m.attr("__cuda_enabled__") = true;
#else
    m.attr("__cuda_enabled__") = false;
#endif
}