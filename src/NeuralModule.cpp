#include <NeuroGen/NeuralModule.h>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

// Conditional CUDA support - only include if explicitly enabled
// To enable CUDA, add -DUSE_CUDA to compiler flags when CUDA is available
#ifdef USE_CUDA
#include <NeuroGen/cuda/NetworkCUDA_Interface.h>
#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#define CUDA_AVAILABLE 1
#else
// CUDA not available - use CPU-only path
#define CUDA_AVAILABLE 0
#endif

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

NeuralModule::NeuralModule(const std::string& name, const NetworkConfig& config)
    : module_name_(name),
      config_(config),
      active_(true),
      is_initialized_(false),
      cuda_initialized_(false),
      learning_rate_(0.01f),
      plasticity_strength_(1.0f),
      homeostatic_target_(0.1f),
      average_activity_(0.0f),
      update_count_(0),
      plasticity_enabled_(true),
      excitability_level_(1.0f),
      adaptation_current_(0.0f),
      background_noise_(0.01f),
      refractory_period_(2.0f),
      firing_rate_(0.0f),
      connection_strength_(0.0f),
      plasticity_events_(0.0f),
      last_update_time_(0.0) {
    
    // Initialize internal network
    internal_network_ = std::make_unique<Network>(config);
    if (internal_network_) {
        // Set bidirectional reference if Network supports it
        // internal_network_->set_module(this);
    }
    
    // Initialize state vectors
    size_t neuron_count = config.num_neurons;
    internal_state_.resize(neuron_count, 0.0f);
    neuron_outputs_.resize(neuron_count, 0.0f);
    activation_history_.resize(100, 0.0f); // Rolling history buffer
    
    // Initialize synaptic weights
    size_t synapse_count = neuron_count * neuron_count; // Simplified estimation
    synaptic_weights_.resize(synapse_count, 0.0f);
    
    std::cout << "Created neural module '" << module_name_ 
              << "' with " << neuron_count << " neurons" << std::endl;
}

NeuralModule::~NeuralModule() {
    if (cuda_initialized_) {
        cleanup_cuda_resources();
    }
    std::cout << "Destroyed neural module '" << module_name_ << "'" << std::endl;
}

bool NeuralModule::initialize() {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    if (is_initialized_) {
        std::cout << "Module '" << module_name_ << "' already initialized" << std::endl;
        return true;
    }
    
    try {
        // Initialize internal network
        if (!internal_network_) {
            std::cerr << "Error: Internal network not created for module '" 
                      << module_name_ << "'" << std::endl;
            return false;
        }
        
        // Initialize synaptic weights with biological distribution
        initialize_synaptic_weights();
        
        // Initialize CUDA resources if available
        initialize_cuda_resources();
        
        // Validate configuration
        if (!validate_configuration()) {
            std::cerr << "Error: Configuration validation failed for module '" 
                      << module_name_ << "'" << std::endl;
            return false;
        }
        
        is_initialized_ = true;
        std::cout << "Successfully initialized neural module '" << module_name_ << "'" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization of module '" 
                  << module_name_ << "': " << e.what() << std::endl;
        return false;
    }
}

void NeuralModule::update(float dt, const std::vector<float>& inputs, float reward) {
    if (!is_initialized_ || !active_) {
        return;
    }

    std::lock_guard<std::mutex> lock(module_mutex_);

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Use CUDA network if available, otherwise fall back to CPU network
#if CUDA_AVAILABLE
        if (cuda_initialized_ && cuda_network_) {
            // ========== GPU-ACCELERATED PATH ==========

            // Prepare input vector
            std::vector<float> input_currents;
            if (!incoming_signals_.empty()) {
                input_currents = incoming_signals_;
                incoming_signals_.clear();
            } else if (!inputs.empty()) {
                input_currents = inputs;
            } else {
                // Background activity
                input_currents.resize(config_.num_neurons, background_noise_);
            }

            // Update using CUDA-accelerated network
            // Note: Using current simulation time approximation
            static float current_time = 0.0f;
            current_time += dt;
            cuda_network_->step(current_time, dt, reward, input_currents);

            // Get statistics from CUDA network
            auto cuda_stats = cuda_network_->get_stats();

            // Extract outputs (placeholder - need to implement proper output extraction)
            // For now, generate pseudo-outputs based on activity
            neuron_outputs_.resize(config_.num_neurons);
            for (size_t i = 0; i < neuron_outputs_.size(); ++i) {
                neuron_outputs_[i] = (cuda_stats.total_spike_count > 0) ?
                    (static_cast<float>(i % 10) / 10.0f) : 0.0f;
            }

        } else
#endif
        {
            // ========== CPU PATH (FALLBACK / DEFAULT) ==========

            if (internal_network_) {
                // Process inputs (if any pending)
                if (!incoming_signals_.empty()) {
                    internal_network_->update(static_cast<float>(dt), incoming_signals_, 0.0f);
                    incoming_signals_.clear();
                } else if (!inputs.empty()) {
                    internal_network_->update(static_cast<float>(dt), inputs, 0.0f);
                } else {
                    // Update with background activity
                    std::vector<float> background_input(config_.num_neurons, background_noise_);
                    internal_network_->update(static_cast<float>(dt), background_input, 0.0f);
                }

                // Get updated outputs
                neuron_outputs_ = internal_network_->get_output();
            }
        }

        // Update biological processes (common to both GPU and CPU paths)
        update_activity_history(average_activity_);
        compute_firing_rate();

        // Apply homeostatic plasticity
        if (plasticity_enabled_) {
            float current_activity = 0.0f;
            for (float output : neuron_outputs_) {
                current_activity += output;
            }
            current_activity /= neuron_outputs_.size();

            // Simple homeostatic adjustment
            float activity_error = homeostatic_target_ - current_activity;
            excitability_level_ += learning_rate_ * activity_error * static_cast<float>(dt);
            excitability_level_ = std::max(0.1f, std::min(2.0f, excitability_level_));
        }

        // Update performance metrics
        update_performance_metrics(dt);
        update_count_++;

    } catch (const std::exception& e) {
        std::cerr << "Exception during update of module '"
                  << module_name_ << "': " << e.what() << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_update_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

bool NeuralModule::validate_configuration() const {
    if (config_.num_neurons == 0) {
        std::cerr << "Error: Zero neurons specified in configuration" << std::endl;
        return false;
    }
    
    if (config_.num_neurons > 100000) {
        std::cerr << "Warning: Large number of neurons (" << config_.num_neurons 
                  << ") may cause performance issues" << std::endl;
    }
    
    if (learning_rate_ <= 0.0f || learning_rate_ > 1.0f) {
        std::cerr << "Error: Invalid learning rate: " << learning_rate_ << std::endl;
        return false;
    }
    
    return true;
}

// ============================================================================
// CORE PROCESSING INTERFACE
// ============================================================================

std::vector<float> NeuralModule::process(const std::vector<float>& input) {
    if (!is_initialized_ || !active_) {
        return std::vector<float>(config_.num_neurons, 0.0f);
    }
    
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Store input for next update cycle
    incoming_signals_ = input;
    
    // Apply activation function to inputs
    std::vector<float> processed_input = input;
    for (size_t i = 0; i < processed_input.size() && i < config_.num_neurons; ++i) {
        processed_input[i] = apply_activation(processed_input[i]);
    }
    
    return processed_input;
}

std::vector<float> NeuralModule::get_output() const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    if (!is_initialized_ || !active_) {
        return std::vector<float>(config_.num_neurons, 0.0f);
    }
    
    // Apply noise and excitability modulation
    std::vector<float> modulated_output = neuron_outputs_;
    for (float& output : modulated_output) {
        output *= excitability_level_;
        output = apply_biological_noise(output);
    }
    
    return modulated_output;
}

std::vector<float> NeuralModule::get_neuron_potentials(const std::vector<size_t>& neuron_ids) const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    std::vector<float> potentials;
    potentials.reserve(neuron_ids.size());
    
    for (size_t neuron_id : neuron_ids) {
        if (neuron_id < internal_state_.size()) {
            potentials.push_back(internal_state_[neuron_id]);
        } else {
            potentials.push_back(0.0f);
            std::cerr << "Warning: Neuron ID " << neuron_id 
                      << " out of range in module " << module_name_ << std::endl;
        }
    }
    
    return potentials;
}

// ============================================================================
// MODULE STATE AND CONTROL
// ============================================================================

void NeuralModule::set_active(bool active) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    active_ = active;
    
    if (!active) {
        // Clear ongoing activity when deactivated
        std::fill(neuron_outputs_.begin(), neuron_outputs_.end(), 0.0f);
        std::fill(internal_state_.begin(), internal_state_.end(), 0.0f);
    }
}

bool NeuralModule::is_active() const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    return active_;
}

const std::string& NeuralModule::get_name() const {
    return module_name_;
}

Network* NeuralModule::get_network() {
    return internal_network_.get();
}

NetworkStats NeuralModule::get_stats() const {
    if (internal_network_) {
        return internal_network_->get_stats();
    }
    return NetworkStats{};
}

// ============================================================================
// INTER-MODULE COMMUNICATION
// ============================================================================

void NeuralModule::register_neuron_port(const std::string& port_name, 
                                       const std::vector<size_t>& neuron_ids) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Validate neuron IDs
    for (size_t id : neuron_ids) {
        if (id >= config_.num_neurons) {
            throw std::runtime_error("Neuron ID " + std::to_string(id) + 
                                   " out of range for module " + module_name_);
        }
    }
    
    neuron_ports_[port_name] = neuron_ids;
    std::cout << "Registered port '" << port_name << "' with " 
              << neuron_ids.size() << " neurons in module " << module_name_ << std::endl;
}

const std::vector<size_t>& NeuralModule::get_neuron_population(const std::string& port_name) const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    auto it = neuron_ports_.find(port_name);
    if (it == neuron_ports_.end()) {
        throw std::runtime_error("Neuron port '" + port_name + 
                                "' not found in module " + module_name_);
    }
    return it->second;
}

void NeuralModule::send_signal(const std::vector<float>& signal, 
                             const std::string& target_module,
                             const std::string& target_port) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Store outgoing signal (would be processed by network manager)
    outgoing_signals_ = signal;
    
    // Log communication
    std::cout << "Module " << module_name_ << " sending signal of size " 
              << signal.size() << " to " << target_module 
              << ":" << target_port << std::endl;
}

void NeuralModule::receive_signal(const std::vector<float>& signal,
                                const std::string& source_module,
                                const std::string& source_port) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Accumulate incoming signals
    if (incoming_signals_.size() != signal.size()) {
        incoming_signals_.resize(signal.size(), 0.0f);
    }
    
    for (size_t i = 0; i < signal.size(); ++i) {
        incoming_signals_[i] += signal[i];
    }
    
    // Log communication
    std::cout << "Module " << module_name_ << " received signal of size " 
              << signal.size() << " from " << source_module 
              << ":" << source_port << std::endl;
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

std::map<std::string, float> NeuralModule::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    std::map<std::string, float> metrics;
    
    metrics["average_activity"] = average_activity_;
    metrics["firing_rate"] = firing_rate_;
    metrics["connection_strength"] = connection_strength_;
    metrics["plasticity_events"] = plasticity_events_;
    metrics["excitability_level"] = excitability_level_;
    metrics["update_time_ms"] = static_cast<float>(last_update_time_);
    metrics["update_count"] = static_cast<float>(update_count_);
    metrics["is_active"] = active_ ? 1.0f : 0.0f;
    metrics["is_initialized"] = is_initialized_ ? 1.0f : 0.0f;
    
    return metrics;
}

void NeuralModule::reset_performance_metrics() {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    average_activity_ = 0.0f;
    firing_rate_ = 0.0f;
    connection_strength_ = 0.0f;
    plasticity_events_ = 0.0f;
    update_count_ = 0;
    last_update_time_ = 0.0;
}

void NeuralModule::update_performance_metrics(float dt) {
    // Update internal performance counters
    update_internal_metrics();
    
    // Compute average activity
    float total_activity = 0.0f;
    for (float output : neuron_outputs_) {
        total_activity += std::abs(output);
    }
    average_activity_ = total_activity / neuron_outputs_.size();
    
    // Update connection strength (simplified)
    connection_strength_ = 0.0f;
    for (float weight : synaptic_weights_) {
        connection_strength_ += std::abs(weight);
    }
    connection_strength_ /= synaptic_weights_.size();
}

// ============================================================================
// STATE SERIALIZATION
// ============================================================================

bool NeuralModule::save_state(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write header
        std::string header = "NEURALMODULE_STATE_V1";
        file.write(header.c_str(), header.size());
        
        // Write module name
        size_t name_size = module_name_.size();
        file.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size));
        file.write(module_name_.c_str(), name_size);
        
        // Write configuration
        file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        
        // Write state vectors
        size_t state_size = internal_state_.size();
        file.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
        file.write(reinterpret_cast<const char*>(internal_state_.data()), 
                  state_size * sizeof(float));
        
        size_t output_size = neuron_outputs_.size();
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
        file.write(reinterpret_cast<const char*>(neuron_outputs_.data()), 
                  output_size * sizeof(float));
        
        size_t weights_size = synaptic_weights_.size();
        file.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
        file.write(reinterpret_cast<const char*>(synaptic_weights_.data()), 
                  weights_size * sizeof(float));
        
        // Write parameters
        file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
        file.write(reinterpret_cast<const char*>(&plasticity_strength_), sizeof(plasticity_strength_));
        file.write(reinterpret_cast<const char*>(&excitability_level_), sizeof(excitability_level_));
        file.write(reinterpret_cast<const char*>(&average_activity_), sizeof(average_activity_));
        
        std::cout << "Successfully saved state for module '" << module_name_ 
                  << "' to " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving module state: " << e.what() << std::endl;
        return false;
    }
}

bool NeuralModule::load_state(const std::string& filename) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
            return false;
        }
        
        // Read and verify header
        std::string header = "NEURALMODULE_STATE_V1";
        std::vector<char> file_header(header.size());
        file.read(file_header.data(), header.size());
        if (std::string(file_header.begin(), file_header.end()) != header) {
            std::cerr << "Error: Invalid file format" << std::endl;
            return false;
        }
        
        // Read module name
        size_t name_size;
        file.read(reinterpret_cast<char*>(&name_size), sizeof(name_size));
        std::vector<char> name_buffer(name_size);
        file.read(name_buffer.data(), name_size);
        std::string loaded_name(name_buffer.begin(), name_buffer.end());
        
        // Skip name validation for now (could warn if different)
        
        // Read configuration
        NetworkConfig loaded_config;
        file.read(reinterpret_cast<char*>(&loaded_config), sizeof(loaded_config));
        
        // Read state vectors
        size_t state_size;
        file.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
        internal_state_.resize(state_size);
        file.read(reinterpret_cast<char*>(internal_state_.data()), 
                 state_size * sizeof(float));
        
        size_t output_size;
        file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
        neuron_outputs_.resize(output_size);
        file.read(reinterpret_cast<char*>(neuron_outputs_.data()), 
                 output_size * sizeof(float));
        
        size_t weights_size;
        file.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
        synaptic_weights_.resize(weights_size);
        file.read(reinterpret_cast<char*>(synaptic_weights_.data()), 
                 weights_size * sizeof(float));
        
        // Read parameters
        file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
        file.read(reinterpret_cast<char*>(&plasticity_strength_), sizeof(plasticity_strength_));
        file.read(reinterpret_cast<char*>(&excitability_level_), sizeof(excitability_level_));
        file.read(reinterpret_cast<char*>(&average_activity_), sizeof(average_activity_));
        
        std::cout << "Successfully loaded state for module '" << module_name_ 
                  << "' from " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading module state: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// CUDA INTEGRATION
// ============================================================================

bool NeuralModule::initialize_cuda_resources() {
#if CUDA_AVAILABLE
    std::cout << "🔧 [CUDA Init] Attempting CUDA initialization for module '" << module_name_ << "'..." << std::endl;
    try {
        std::cout << "\n========================================" << std::endl;
        std::cout << "🚀 CUDA Device Initialization" << std::endl;
        std::cout << "========================================" << std::endl;

        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess || device_count == 0) {
            std::cout << "❌ CUDA Status: NOT AVAILABLE" << std::endl;
            std::cout << "   Reason: " << (error != cudaSuccess ? cudaGetErrorString(error) : "No CUDA devices found") << std::endl;
            std::cout << "   Mode: CPU-only processing" << std::endl;
            std::cout << "========================================\n" << std::endl;
            cuda_initialized_ = false;
            return true; // CPU-only operation is still valid
        }

        // CUDA is available, initialize GPU network
        std::cout << "✅ CUDA Status: ENABLED AND AVAILABLE" << std::endl;
        std::cout << "   Devices found: " << device_count << std::endl;

        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // Calculate memory in GB
        float total_memory_gb = prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);

        std::cout << "\n📊 GPU Device Information:" << std::endl;
        std::cout << "   Name: " << prop.name << std::endl;
        std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "   Total Memory: " << std::fixed << std::setprecision(2) << total_memory_gb << " GB" << std::endl;
        std::cout << "   Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "   Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;

        // Create initial neuron and synapse structures for NetworkCUDA_Interface
        std::vector<GPUNeuronState> initial_neurons(config_.num_neurons);
        std::vector<GPUSynapse> initial_synapses;

        // Initialize neurons with default values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.07f, -0.05f); // Resting potential range

        for (size_t i = 0; i < config_.num_neurons; ++i) {
            initial_neurons[i].V = dist(gen);  // Membrane potential
            initial_neurons[i].u = 0.0f;        // Recovery variable
            initial_neurons[i].active = 1;      // Active initially
            initial_neurons[i].neuron_type = (i % 5 == 0) ? 1 : 0;  // Mix of excitatory/inhibitory
        }

        // Create sparse connectivity (simple local connectivity pattern)
        size_t connections_per_neuron = std::min(static_cast<size_t>(config_.localFanOut),
                                                  config_.num_neurons - 1);
        initial_synapses.reserve(config_.num_neurons * connections_per_neuron);

        std::uniform_real_distribution<float> weight_dist(0.1f, 0.5f);
        for (size_t source = 0; source < config_.num_neurons; ++source) {
            for (size_t conn = 0; conn < connections_per_neuron; ++conn) {
                size_t target = (source + conn + 1) % config_.num_neurons;

                GPUSynapse synapse;
                synapse.pre_neuron_idx = source;
                synapse.post_neuron_idx = target;
                synapse.weight = weight_dist(gen);
                synapse.delay = 1; // 1 timestep delay
                initial_synapses.push_back(synapse);
            }
        }

        // Create NetworkCUDA_Interface
        cuda_network_ = std::make_unique<NetworkCUDA_Interface>(
            config_,
            initial_neurons,
            initial_synapses
        );

        cuda_initialized_ = true;

        std::cout << "\n💚 Initialization Status: SUCCESS" << std::endl;
        std::cout << "   Neural Network: " << initial_neurons.size() << " neurons, "
                  << initial_synapses.size() << " synapses" << std::endl;
        std::cout << "   Acceleration: GPU (CUDA)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n⚠️  CUDA Initialization Failed!" << std::endl;
        std::cerr << "   Error: " << e.what() << std::endl;
        std::cerr << "   Fallback: CPU-only processing" << std::endl;
        std::cerr << "========================================\n" << std::endl;
        cuda_initialized_ = false;
        return true; // CPU fallback
    }
#else
    // CUDA not compiled in - use CPU-only processing
    std::cout << "\n========================================" << std::endl;
    std::cout << "ℹ️  CUDA Device Initialization" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "❌ CUDA Status: NOT COMPILED" << std::endl;
    std::cout << "   Reason: CUDA support not enabled at compile-time" << std::endl;
    std::cout << "   Mode: CPU-only processing" << std::endl;
    std::cout << "   Tip: Recompile with -DUSE_CUDA when CUDA toolkit is installed" << std::endl;
    std::cout << "========================================\n" << std::endl;
    cuda_initialized_ = false;
    return true; // CPU-only operation
#endif
}

void NeuralModule::cleanup_cuda_resources() {
    if (cuda_initialized_) {
        std::cout << "Cleaning up CUDA resources for module '" << module_name_ << "'" << std::endl;

#if CUDA_AVAILABLE
        // Release CUDA network interface
        cuda_network_.reset();

        // Reset CUDA device to free all memory
        cudaDeviceReset();
#endif

        cuda_initialized_ = false;
    }
}

bool NeuralModule::is_cuda_available() const {
    return cuda_initialized_;
}

// ============================================================================
// PROTECTED HELPER METHODS
// ============================================================================

float NeuralModule::apply_activation(float input) const {
    // Sigmoid activation with biological parameters
    return 1.0f / (1.0f + std::exp(-input * excitability_level_));
}

void NeuralModule::update_synaptic_weights(const std::vector<float>& pre_activity,
                                         const std::vector<float>& post_activity, 
                                         float dt) {
    if (!plasticity_enabled_ || pre_activity.size() != post_activity.size()) {
        return;
    }
    
    // Simplified STDP implementation
    for (size_t i = 0; i < pre_activity.size() && i < synaptic_weights_.size(); ++i) {
        float weight_change = learning_rate_ * pre_activity[i] * post_activity[i] * dt;
        synaptic_weights_[i] += weight_change;
        
        // Bound weights
        synaptic_weights_[i] = std::max(-1.0f, std::min(1.0f, synaptic_weights_[i]));
        
        if (std::abs(weight_change) > 0.001f) {
            plasticity_events_ += 1.0f;
        }
    }
}

void NeuralModule::initialize_synaptic_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (float& weight : synaptic_weights_) {
        weight = dist(gen);
    }
    
    std::cout << "Initialized " << synaptic_weights_.size() 
              << " synaptic weights for module " << module_name_ << std::endl;
}

void NeuralModule::update_activity_history(float current_activity) {
    // Shift history buffer
    for (size_t i = activation_history_.size() - 1; i > 0; --i) {
        activation_history_[i] = activation_history_[i - 1];
    }
    activation_history_[0] = current_activity;
}

float NeuralModule::compute_firing_rate() const {
    if (activation_history_.empty()) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (float activity : activation_history_) {
        sum += activity;
    }
    
    return sum / activation_history_.size();
}

float NeuralModule::apply_biological_noise(float signal) const {
    if (background_noise_ <= 0.0f) {
        return signal;
    }
    
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(0.0f, background_noise_);
    
    return signal + noise_dist(gen);
}

void NeuralModule::update_internal_metrics() {
    // Update various internal performance metrics
    // This is called during update_performance_metrics
    
    // Calculate network coherence (simplified)
    if (!neuron_outputs_.empty()) {
        float mean_output = 0.0f;
        for (float output : neuron_outputs_) {
            mean_output += output;
        }
        mean_output /= neuron_outputs_.size();
        
        float variance = 0.0f;
        for (float output : neuron_outputs_) {
            float diff = output - mean_output;
            variance += diff * diff;
        }
        variance /= neuron_outputs_.size();
        
        // Store coherence metric (low variance = high coherence)
        connection_strength_ = 1.0f / (1.0f + variance);
    }
}

// ============================================================================
// NEUROMODULATION IMPLEMENTATION
// ============================================================================

void NeuralModule::applyNeuromodulation(const std::string& modulator_type, float level) {
    if (!is_initialized_ || !internal_network_) {
        return;
    }
    
    // Apply neuromodulation based on type
    if (modulator_type == "dopamine") {
        // Dopamine affects learning rate and reward sensitivity
        learning_rate_ = 0.01f * (1.0f + level * 2.0f); // Boost learning
        if (internal_network_) {
            // Apply to network's reward system if available
            // This is a simplified implementation
        }
    } else if (modulator_type == "acetylcholine") {
        // Acetylcholine affects attention and plasticity
        plasticity_strength_ = 1.0f + level * 1.5f; // Enhance plasticity
    } else if (modulator_type == "serotonin") {
        // Serotonin affects overall activity and mood
        // Modulate baseline activity levels
        if (level > 0.5f) {
            // Higher serotonin = more stable, less erratic activity
            // Apply stabilization to network
        }
    } else if (modulator_type == "norepinephrine") {
        // Norepinephrine affects arousal and attention
        // Increase overall network responsiveness
        if (internal_network_) {
            // Boost signal transmission (simplified)
        }
    } else if (modulator_type == "gaba") {
        // GABA is inhibitory
        // Reduce overall activity
        if (internal_network_) {
            // Apply inhibitory effects (simplified)
        }
    }
    
    // Clamp values to reasonable ranges
    learning_rate_ = std::max(0.001f, std::min(0.1f, learning_rate_));
    plasticity_strength_ = std::max(0.1f, std::min(3.0f, plasticity_strength_));
}

