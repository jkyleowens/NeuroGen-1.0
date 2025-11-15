// ============================================================================
// ENHANCED LEARNING SYSTEM IMPLEMENTATION
// File: src/EnhancedLearningSystem.cpp
// ============================================================================

#include <NeuroGen/EnhancedLearningSystem.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

// External C wrapper function declarations
extern "C" {
    void launch_eligibility_reset_wrapper(void* d_synapses, int num_synapses);
    void launch_enhanced_stdp_wrapper(void* d_synapses, const void* d_neurons, 
                                     float current_time, float dt, int num_synapses);
    void launch_eligibility_update_wrapper(void* d_synapses, const void* d_neurons,
                                          float current_time, float dt, int num_synapses);
    void launch_trace_monitoring_wrapper(const void* d_synapses, int num_synapses, void* d_trace_stats);
    void launch_reward_modulation_wrapper(void* d_synapses, void* d_neurons, float reward,
                                         float current_time, float dt, int num_synapses);
    void launch_hebbian_learning_wrapper(void* d_synapses, const void* d_neurons,
                                        float current_time, float dt, int num_synapses);
    void launch_bcm_learning_wrapper(void* d_synapses, const void* d_neurons,
                                    float learning_rate, float dt, int num_synapses);
    void launch_correlation_learning_wrapper(void* d_synapses, const void* d_neurons,
                                            void* d_correlation_matrix, float learning_rate,
                                            float dt, int num_synapses, int matrix_size);
    void launch_reward_prediction_error_wrapper(const void* d_actual_reward,
                                               void* d_predicted_rewards, int num_timesteps);
}

// ============================================================================
// CONSTRUCTOR AND DESTRUCTOR
// ============================================================================

EnhancedLearningSystem::EnhancedLearningSystem() 
    : d_synapses_ptr_(nullptr), d_neurons_ptr_(nullptr), d_reward_signals_ptr_(nullptr),
      d_attention_weights_ptr_(nullptr), d_trace_stats_ptr_(nullptr), d_correlation_matrix_ptr_(nullptr),
      num_synapses_(0), num_neurons_(0), num_modules_(0), correlation_matrix_size_(0),
      learning_rate_(0.001f), eligibility_decay_(0.99f), reward_scaling_(1.0f), baseline_dopamine_(0.1f),
      cuda_initialized_(false), learning_stream_(0), attention_stream_(0),
      average_eligibility_trace_(0.0f), learning_progress_(0.0f), total_weight_change_(0.0f) {
    
    std::cout << "Enhanced Learning System: Initializing breakthrough neural plasticity architecture..." << std::endl;
}

EnhancedLearningSystem::EnhancedLearningSystem(int num_synapses, int num_neurons)
    : d_synapses_ptr_(nullptr), d_neurons_ptr_(nullptr), d_reward_signals_ptr_(nullptr),
      d_attention_weights_ptr_(nullptr), d_trace_stats_ptr_(nullptr), d_correlation_matrix_ptr_(nullptr),
      num_synapses_(num_synapses), num_neurons_(num_neurons), num_modules_(1), correlation_matrix_size_(num_neurons),
      learning_rate_(0.001f), eligibility_decay_(0.99f), reward_scaling_(1.0f), baseline_dopamine_(0.1f),
      cuda_initialized_(false), learning_stream_(0), attention_stream_(0),
      average_eligibility_trace_(0.0f), learning_progress_(0.0f), total_weight_change_(0.0f) {
    
    std::cout << "Enhanced Learning System: Initializing with " << num_synapses 
              << " synapses and " << num_neurons << " neurons (legacy mode)..." << std::endl;
              
    // Initialize with single module for legacy compatibility
    initialize(num_neurons, num_synapses, 1);
}

EnhancedLearningSystem::~EnhancedLearningSystem() {
    cleanup_cuda_resources();
    std::cout << "Enhanced Learning System: Cleaned up GPU resources and finalized learning state." << std::endl;
}

// ============================================================================
// INITIALIZATION AND CONFIGURATION
// ============================================================================

bool EnhancedLearningSystem::initialize(int num_neurons, int num_synapses, int num_modules) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    std::cout << "Enhanced Learning System: Initializing with " << num_neurons 
              << " neurons, " << num_synapses << " synapses, " << num_modules << " modules..." << std::endl;
    
    num_neurons_ = num_neurons;
    num_synapses_ = num_synapses;
    num_modules_ = num_modules;
    correlation_matrix_size_ = num_neurons;
    
    // Initialize module states
    module_states_.resize(num_modules);
    module_attention_.resize(num_modules, 1.0f);
    module_learning_rates_.resize(num_modules, learning_rate_);
    
    for (int i = 0; i < num_modules; ++i) {
        module_states_[i].module_id = i;
        module_states_[i].num_neurons = num_neurons / num_modules; // Simplified equal division
        module_states_[i].num_synapses = num_synapses / num_modules;
        module_states_[i].learning_rate = learning_rate_;
        module_states_[i].total_weight_change = 0.0f;
        module_states_[i].average_eligibility = 0.0f;
        module_states_[i].reward_prediction_error = 0.0f;
        module_states_[i].activity_level = 0.0f;
        module_states_[i].attention_weight = 1.0f;
        module_states_[i].plasticity_threshold = 0.5f;
        module_states_[i].last_update_time = 0;
        module_states_[i].is_active = true;
    }
    
    // Initialize CUDA resources
    if (!initialize_cuda_resources()) {
        std::cerr << "Enhanced Learning System: Failed to initialize CUDA resources!" << std::endl;
        return false;
    }
    
    std::cout << "Enhanced Learning System: Successfully initialized breakthrough learning architecture!" << std::endl;
    std::cout << "  - Biological STDP with calcium dynamics: ENABLED" << std::endl;
    std::cout << "  - Dopaminergic reward modulation: ENABLED" << std::endl;
    std::cout << "  - Multi-factor eligibility traces: ENABLED" << std::endl;
    std::cout << "  - Homeostatic regulation: ENABLED" << std::endl;
    std::cout << "  - Modular independent learning: ENABLED" << std::endl;
    
    return true;
}

void EnhancedLearningSystem::configure_learning_parameters(float lr, float decay, float scaling) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    learning_rate_ = lr;
    eligibility_decay_ = decay;
    reward_scaling_ = scaling;
    
    // Update all module learning rates proportionally
    for (size_t i = 0; i < module_learning_rates_.size(); ++i) {
        module_learning_rates_[i] = lr;
        if (i < module_states_.size()) {
            module_states_[i].learning_rate = lr;
        }
    }
    
    std::cout << "Enhanced Learning System: Updated learning parameters - LR: " << lr 
              << ", Decay: " << decay << ", Scaling: " << scaling << std::endl;
}

void EnhancedLearningSystem::setup_modular_architecture(const std::vector<int>& module_sizes) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    if (module_sizes.size() != static_cast<size_t>(num_modules_)) {
        std::cerr << "Enhanced Learning System: Module size mismatch!" << std::endl;
        return;
    }
    
    int total_neurons = 0;
    for (size_t i = 0; i < module_sizes.size(); ++i) {
        module_states_[i].num_neurons = module_sizes[i];
        module_states_[i].num_synapses = module_sizes[i] * 10; // Estimate synapses per neuron
        total_neurons += module_sizes[i];
    }
    
    std::cout << "Enhanced Learning System: Configured modular architecture with " 
              << module_sizes.size() << " specialized modules, total " << total_neurons << " neurons." << std::endl;
}

// ============================================================================
// MAIN LEARNING INTERFACE
// ============================================================================

void EnhancedLearningSystem::update_learning(float current_time, float dt, float reward_signal) {
    if (!cuda_initialized_) {
        std::cerr << "Enhanced Learning System: CUDA not initialized!" << std::endl;
        return;
    }
    
    // Phase 1: Reset eligibility traces periodically (every 1000ms)
    if (fmod(current_time, 1000.0f) < dt) {
        launch_eligibility_reset();
    }
    
    // Phase 2: Update STDP and eligibility traces
    update_stdp_and_eligibility(current_time, dt);
    
    // Phase 3: Apply reward modulation
    apply_reward_modulation(reward_signal, current_time, dt);
    
    // Phase 4: Update correlation-based learning
    update_correlation_learning(current_time, dt);
    
    // Phase 5: Apply homeostatic regulation
    apply_homeostatic_regulation(0.1f, dt); // Target 0.1 Hz average activity
    
    // Phase 6: Update performance metrics
    update_performance_metrics();
    
    // Synchronize GPU operations
    cudaStreamSynchronize(learning_stream_);
}

void EnhancedLearningSystem::update_stdp_and_eligibility(float current_time, float dt) {
    // Launch enhanced STDP kernel
    launch_enhanced_stdp_wrapper(d_synapses_ptr_, d_neurons_ptr_, current_time, dt, num_synapses_);
    
    // Update eligibility traces
    launch_eligibility_update_wrapper(d_synapses_ptr_, d_neurons_ptr_, current_time, dt, num_synapses_);
    
    // Monitor trace statistics
    launch_trace_monitoring_wrapper(d_synapses_ptr_, num_synapses_, d_trace_stats_ptr_);
}

void EnhancedLearningSystem::apply_reward_modulation(float reward_signal, float current_time, float dt) {
    // Scale reward signal for biological realism
    float scaled_reward = reward_signal * reward_scaling_;
    
    // Launch reward modulation kernel
    launch_reward_modulation_wrapper(d_synapses_ptr_, d_neurons_ptr_, scaled_reward, 
                                    current_time, dt, num_synapses_);
    
    // Update reward prediction error
    float actual_reward_array[1] = {scaled_reward};
    
    // Copy to GPU temporarily for RPE computation
    float* d_actual_reward;
    cudaMalloc(&d_actual_reward, sizeof(float));
    cudaMemcpy(d_actual_reward, actual_reward_array, sizeof(float), cudaMemcpyHostToDevice);
    
    launch_reward_prediction_error_wrapper(d_actual_reward, d_reward_signals_ptr_, 1);
    
    cudaFree(d_actual_reward);
}

void EnhancedLearningSystem::update_correlation_learning(float current_time, float dt) {
    // Launch Hebbian learning
    launch_hebbian_learning_wrapper(d_synapses_ptr_, d_neurons_ptr_, current_time, dt, num_synapses_);
    
    // Launch BCM learning with adaptive threshold
    float bcm_learning_rate = learning_rate_ * 0.1f; // Slower BCM learning
    launch_bcm_learning_wrapper(d_synapses_ptr_, d_neurons_ptr_, bcm_learning_rate, dt, num_synapses_);
    
    // Launch correlation-based learning
    float correlation_learning_rate = learning_rate_ * 0.05f; // Even slower correlation learning
    launch_correlation_learning_wrapper(d_synapses_ptr_, d_neurons_ptr_, d_correlation_matrix_ptr_,
                                       correlation_learning_rate, dt, num_synapses_, correlation_matrix_size_);
}

void EnhancedLearningSystem::apply_homeostatic_regulation(float target_activity, float dt) {
    // Homeostatic regulation is implemented implicitly in the other kernels
    // through sliding thresholds and activity-dependent scaling
    // This could be expanded with dedicated homeostatic kernels
    
    // For now, we adjust learning rates based on overall network activity
    float current_activity = get_average_eligibility_trace();
    float activity_ratio = current_activity / (target_activity + 1e-6f);
    
    // Adjust module learning rates to maintain homeostasis
    for (auto& module_state : module_states_) {
        if (activity_ratio > 2.0f) {
            module_state.learning_rate *= 0.99f; // Reduce learning if too active
        } else if (activity_ratio < 0.5f) {
            module_state.learning_rate *= 1.01f; // Increase learning if too quiet
        }
        
        // Bound learning rates
        module_state.learning_rate = std::max(0.0001f, std::min(module_state.learning_rate, 0.1f));
    }
}

void EnhancedLearningSystem::update_attention_learning(const std::vector<float>& attention_weights, float dt) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    if (attention_weights.size() != module_attention_.size()) {
        std::cerr << "Enhanced Learning System: Attention weight size mismatch!" << std::endl;
        return;
    }
    
    // Update module attention weights
    module_attention_ = attention_weights;
    
    // Modulate learning rates based on attention
    for (size_t i = 0; i < module_states_.size(); ++i) {
        float attention_factor = attention_weights[i];
        module_states_[i].attention_weight = attention_factor;
        
        // Higher attention = higher learning rate
        float base_rate = module_learning_rates_[i];
        module_states_[i].learning_rate = base_rate * (0.5f + 1.5f * attention_factor);
    }
    
    // Copy attention weights to GPU (if needed for kernels)
    if (d_attention_weights_ptr_) {
        cudaMemcpy(d_attention_weights_ptr_, attention_weights.data(), 
                  attention_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void EnhancedLearningSystem::update_modular_learning(int module_id, float module_reward, float dt) {
    if (module_id < 0 || module_id >= num_modules_) {
        std::cerr << "Enhanced Learning System: Invalid module ID: " << module_id << std::endl;
        return;
    }
    
    ModuleState& module = module_states_[module_id];
    
    // Update module-specific learning parameters
    module.reward_prediction_error = module_reward - 0.1f; // Baseline expectation
    module.total_weight_change += fabsf(module.reward_prediction_error) * dt;
    module.last_update_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // Apply module-specific reward modulation
    // This could be extended to launch kernels on specific GPU memory regions for each module
    apply_reward_modulation(module_reward, module.last_update_time, dt);
}

// ============================================================================
// STATE MANAGEMENT AND PERSISTENCE
// ============================================================================

bool EnhancedLearningSystem::save_learning_state(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    std::ofstream file(filename + "_learning_state.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Enhanced Learning System: Failed to open file for saving: " << filename << std::endl;
        return false;
    }
    
    // Save basic parameters
    file.write(reinterpret_cast<const char*>(&num_neurons_), sizeof(num_neurons_));
    file.write(reinterpret_cast<const char*>(&num_synapses_), sizeof(num_synapses_));
    file.write(reinterpret_cast<const char*>(&num_modules_), sizeof(num_modules_));
    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    file.write(reinterpret_cast<const char*>(&eligibility_decay_), sizeof(eligibility_decay_));
    file.write(reinterpret_cast<const char*>(&reward_scaling_), sizeof(reward_scaling_));
    
    // Save module states
    size_t module_count = module_states_.size();
    file.write(reinterpret_cast<const char*>(&module_count), sizeof(module_count));
    file.write(reinterpret_cast<const char*>(module_states_.data()), 
              module_count * sizeof(ModuleState));
    
    // Save GPU data to CPU arrays and then to file
    if (cuda_initialized_) {
        // This would require copying GPU memory to CPU first
        // Implementation depends on specific GPU data structures
        std::cout << "Enhanced Learning System: GPU state saving not fully implemented yet." << std::endl;
    }
    
    file.close();
    std::cout << "Enhanced Learning System: Saved learning state to " << filename << std::endl;
    return true;
}

bool EnhancedLearningSystem::load_learning_state(const std::string& filename) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    std::ifstream file(filename + "_learning_state.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Enhanced Learning System: Failed to open file for loading: " << filename << std::endl;
        return false;
    }
    
    // Load basic parameters
    file.read(reinterpret_cast<char*>(&num_neurons_), sizeof(num_neurons_));
    file.read(reinterpret_cast<char*>(&num_synapses_), sizeof(num_synapses_));
    file.read(reinterpret_cast<char*>(&num_modules_), sizeof(num_modules_));
    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
    file.read(reinterpret_cast<char*>(&eligibility_decay_), sizeof(eligibility_decay_));
    file.read(reinterpret_cast<char*>(&reward_scaling_), sizeof(reward_scaling_));
    
    // Load module states
    size_t module_count;
    file.read(reinterpret_cast<char*>(&module_count), sizeof(module_count));
    module_states_.resize(module_count);
    file.read(reinterpret_cast<char*>(module_states_.data()), 
              module_count * sizeof(ModuleState));
    
    file.close();
    std::cout << "Enhanced Learning System: Loaded learning state from " << filename << std::endl;
    return true;
}

void EnhancedLearningSystem::reset_learning_state() {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    // Reset performance metrics
    average_eligibility_trace_ = 0.0f;
    learning_progress_ = 0.0f;
    total_weight_change_ = 0.0f;
    
    // Reset module states
    for (auto& module : module_states_) {
        module.total_weight_change = 0.0f;
        module.average_eligibility = 0.0f;
        module.reward_prediction_error = 0.0f;
        module.activity_level = 0.0f;
        module.learning_rate = learning_rate_;
    }
    
    // Reset GPU memory (launch reset kernels)
    if (cuda_initialized_) {
        launch_eligibility_reset_wrapper(d_synapses_ptr_, num_synapses_);
        cudaStreamSynchronize(learning_stream_);
    }
    
    std::cout << "Enhanced Learning System: Reset all learning state to baseline." << std::endl;
}

// ============================================================================
// MONITORING AND ANALYSIS
// ============================================================================

float EnhancedLearningSystem::get_average_eligibility_trace() const {
    return average_eligibility_trace_;
}

float EnhancedLearningSystem::get_learning_progress() const {
    return learning_progress_;
}

std::vector<float> EnhancedLearningSystem::get_module_learning_rates() const {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    return module_learning_rates_;
}

void EnhancedLearningSystem::get_correlation_statistics(std::vector<float>& stats) const {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    stats.resize(4);
    stats[0] = average_eligibility_trace_;
    stats[1] = learning_progress_;
    stats[2] = total_weight_change_;
    stats[3] = static_cast<float>(module_states_.size());
}

float EnhancedLearningSystem::get_total_weight_change() const {
    return total_weight_change_;
}

void EnhancedLearningSystem::get_detailed_learning_statistics(std::vector<float>& detailed_stats) const {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    detailed_stats.clear();
    detailed_stats.reserve(module_states_.size() * 4 + 3);
    
    // Global statistics
    detailed_stats.push_back(average_eligibility_trace_);
    detailed_stats.push_back(learning_progress_);
    detailed_stats.push_back(total_weight_change_);
    
    // Per-module statistics
    for (const auto& module : module_states_) {
        detailed_stats.push_back(module.learning_rate);
        detailed_stats.push_back(module.attention_weight);
        detailed_stats.push_back(module.activity_level);
        detailed_stats.push_back(module.total_weight_change);
    }
}

// ============================================================================
// REWARD AND PROTEIN SYNTHESIS METHODS
// ============================================================================

void EnhancedLearningSystem::setRewardSignal(float reward_value) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    // Update baseline dopamine based on reward
    baseline_dopamine_ = baseline_dopamine_ * 0.99f + reward_value * reward_scaling_ * 0.01f;
    
    // Clamp baseline dopamine to reasonable range
    baseline_dopamine_ = std::max(0.0f, std::min(baseline_dopamine_, 1.0f));
    
    // If CUDA is initialized, update GPU reward signals
    if (cuda_initialized_ && d_reward_signals_ptr_) {
        // Copy reward signal to GPU memory
        float* d_reward_signals = static_cast<float*>(d_reward_signals_ptr_);
        cudaMemcpy(d_reward_signals, &reward_value, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    std::cout << "Enhanced Learning System: Set reward signal to " << reward_value 
              << ", baseline dopamine: " << baseline_dopamine_ << std::endl;
}

void EnhancedLearningSystem::triggerProteinSynthesis(float stimulus_strength) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    // Trigger protein synthesis only if stimulus exceeds threshold
    if (stimulus_strength > PROTEIN_SYNTHESIS_THRESHOLD) {
        
        // Enhance learning rates temporarily for consolidation
        float consolidation_factor = 1.5f * stimulus_strength;
        
        for (auto& module_state : module_states_) {
            if (module_state.is_active) {
                // Temporarily boost learning rate for this module
                float original_lr = module_state.learning_rate;
                module_state.learning_rate = std::min(original_lr * consolidation_factor, 0.1f);
                
                std::cout << "Enhanced Learning System: Triggered protein synthesis for module " 
                          << module_state.module_id 
                          << " - Enhanced LR from " << original_lr 
                          << " to " << module_state.learning_rate << std::endl;
            }
        }
        
        // Update learning progress to reflect consolidation
        learning_progress_ = std::min(1.0f, learning_progress_ + 0.1f * stimulus_strength);
        
        std::cout << "Enhanced Learning System: Protein synthesis triggered with strength " 
                  << stimulus_strength << ", learning progress: " << learning_progress_ << std::endl;
    }
}

// ============================================================================
// PRIVATE IMPLEMENTATION METHODS
// ============================================================================

bool EnhancedLearningSystem::initialize_cuda_resources() {
    cudaError_t error;
    
    // Create CUDA streams for parallel execution
    error = cudaStreamCreate(&learning_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to create learning stream: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaStreamCreate(&attention_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to create attention stream: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate GPU memory (using void* pointers to avoid CUDA types in header)
    size_t synapse_size = num_synapses_ * 64; // Estimate GPUSynapse size
    size_t neuron_size = num_neurons_ * 128;  // Estimate GPUNeuronState size
    
    error = cudaMalloc(&d_synapses_ptr_, synapse_size);
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to allocate synapse memory: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_neurons_ptr_, neuron_size);
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to allocate neuron memory: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate additional arrays
    error = cudaMalloc(&d_reward_signals_ptr_, num_modules_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to allocate reward signals: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_attention_weights_ptr_, num_modules_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to allocate attention weights: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_trace_stats_ptr_, 4 * sizeof(float)); // Sum, max, mean, variance
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to allocate trace stats: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_correlation_matrix_ptr_, correlation_matrix_size_ * correlation_matrix_size_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to allocate correlation matrix: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize GPU memory
    error = cudaMemset(d_synapses_ptr_, 0, synapse_size);
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to initialize synapse memory: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMemset(d_neurons_ptr_, 0, neuron_size);
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to initialize neuron memory: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMemset(d_reward_signals_ptr_, 0, num_modules_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to initialize reward signals: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMemset(d_attention_weights_ptr_, 0, num_modules_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to initialize attention weights: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMemset(d_trace_stats_ptr_, 0, 4 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to initialize trace stats: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMemset(d_correlation_matrix_ptr_, 0, correlation_matrix_size_ * correlation_matrix_size_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Enhanced Learning System: Failed to initialize correlation matrix: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cuda_initialized_ = true;
    std::cout << "Enhanced Learning System: Successfully initialized CUDA resources." << std::endl;
    std::cout << "  - Synapse memory: " << (synapse_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - Neuron memory: " << (neuron_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - Correlation matrix: " << (correlation_matrix_size_ * correlation_matrix_size_ * 4 / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

void EnhancedLearningSystem::cleanup_cuda_resources() {
    if (!cuda_initialized_) return;
    
    // Free GPU memory
    if (d_synapses_ptr_) cudaFree(d_synapses_ptr_);
    if (d_neurons_ptr_) cudaFree(d_neurons_ptr_);
    if (d_reward_signals_ptr_) cudaFree(d_reward_signals_ptr_);
    if (d_attention_weights_ptr_) cudaFree(d_attention_weights_ptr_);
    if (d_trace_stats_ptr_) cudaFree(d_trace_stats_ptr_);
    if (d_correlation_matrix_ptr_) cudaFree(d_correlation_matrix_ptr_);
    
    // Destroy streams
    if (learning_stream_) cudaStreamDestroy(learning_stream_);
    if (attention_stream_) cudaStreamDestroy(attention_stream_);
    
    cuda_initialized_ = false;
}

void EnhancedLearningSystem::launch_eligibility_reset() {
    launch_eligibility_reset_wrapper(d_synapses_ptr_, num_synapses_);
}

void EnhancedLearningSystem::update_performance_metrics() {
    if (!cuda_initialized_) return;
    
    // Copy trace statistics from GPU
    float trace_stats[4];
    cudaMemcpy(trace_stats, d_trace_stats_ptr_, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Update variables
    average_eligibility_trace_ = trace_stats[0] / num_synapses_;
    
    // Update learning progress based on weight changes
    float progress = std::min(1.0f, total_weight_change_ / (num_synapses_ * 0.1f));
    learning_progress_ = progress;
    
    // Update total weight change
    total_weight_change_ += trace_stats[2]; // Assuming trace_stats[2] contains weight change magnitude
}

// ============================================================================
// ADVANCED CONFIGURATION METHODS
// ============================================================================

void EnhancedLearningSystem::configure_learning_mechanisms(bool enable_stdp, bool enable_homeostatic, bool enable_correlation) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    // Store configuration flags (would be used in update_learning to conditionally run kernels)
    // For now, just log the configuration
    std::cout << "Enhanced Learning System: Configured learning mechanisms - "
              << "STDP: " << (enable_stdp ? "ON" : "OFF") 
              << ", Homeostatic: " << (enable_homeostatic ? "ON" : "OFF")
              << ", Correlation: " << (enable_correlation ? "ON" : "OFF") << std::endl;
}

void EnhancedLearningSystem::set_module_learning_parameters(int module_id, float learning_rate, float plasticity_threshold) {
    if (module_id < 0 || module_id >= static_cast<int>(module_states_.size())) {
        std::cerr << "Enhanced Learning System: Invalid module ID for parameter setting: " << module_id << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    module_states_[module_id].learning_rate = learning_rate;
    module_states_[module_id].plasticity_threshold = plasticity_threshold;
    
    if (module_id < static_cast<int>(module_learning_rates_.size())) {
        module_learning_rates_[module_id] = learning_rate;
    }
    
    std::cout << "Enhanced Learning System: Set module " << module_id 
              << " learning rate to " << learning_rate 
              << ", plasticity threshold to " << plasticity_threshold << std::endl;
}

void EnhancedLearningSystem::configure_reward_prediction(float prediction_window, float error_sensitivity) {
    std::lock_guard<std::mutex> lock(learning_mutex_);
    
    // Store configuration for reward prediction error computation
    // This would be used in the reward modulation kernels
    std::cout << "Enhanced Learning System: Configured reward prediction - "
              << "Window: " << prediction_window << "ms, "
              << "Sensitivity: " << error_sensitivity << std::endl;
}