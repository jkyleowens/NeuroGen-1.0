#ifndef ENHANCED_LEARNING_SYSTEM_H
#define ENHANCED_LEARNING_SYSTEM_H

// Core NeuroGen includes with proper paths
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/GridBlockUtils.cuh>

// Enhanced learning rule components
#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/EnhancedSTDPKernel.cuh>
#include <NeuroGen/cuda/EligibilityAndRewardKernels.cuh>
#include <NeuroGen/cuda/RewardModulationKernel.cuh>
#include <NeuroGen/cuda/HebbianLearningKernel.cuh>
#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <mutex>

// Enhanced learning system constants
#define BASELINE_DOPAMINE 0.4f
#define UPDATE_FREQUENCY 50.0f    // ms
#define TARGET_ACTIVITY_LEVEL 0.1f
#define PROTEIN_SYNTHESIS_THRESHOLD 0.8f

/**
 * Enhanced Learning System Manager
 * Coordinates multiple biologically-inspired learning mechanisms
 * in a neurobiologically realistic manner for modular neural networks
 */
class EnhancedLearningSystem {
private:
    // GPU memory pointers (using void* to avoid CUDA types in header)
    void* d_synapses_ptr_;
    void* d_neurons_ptr_;
    void* d_reward_signals_ptr_;
    void* d_attention_weights_ptr_;
    void* d_trace_stats_ptr_;
    void* d_correlation_matrix_ptr_;
    
    // Network parameters
    int num_synapses_;
    int num_neurons_;
    int num_modules_;
    int correlation_matrix_size_;
    
    // Learning parameters
    float learning_rate_;
    float eligibility_decay_;
    float reward_scaling_;
    float baseline_dopamine_;
    
    // CUDA resources
    bool cuda_initialized_;
    cudaStream_t learning_stream_;
    cudaStream_t attention_stream_;
    
    // Module state tracking
    struct ModuleState {
        int module_id;
        int num_neurons;
        int num_synapses;
        float learning_rate;
        float total_weight_change;
        float average_eligibility;
        float reward_prediction_error;
        float activity_level;
        float attention_weight;
        float plasticity_threshold;
        long long last_update_time;
        bool is_active;
    };
    
    std::vector<ModuleState> module_states_;
    std::vector<float> module_attention_;
    std::vector<float> module_learning_rates_;
    
    // Performance monitoring
    float average_eligibility_trace_;
    float learning_progress_;
    float total_weight_change_;
    
    // Thread safety
    mutable std::mutex learning_mutex_;

public:
    // ====================================================================
    // CONSTRUCTORS AND DESTRUCTORS
    // ====================================================================
    
    /**
     * Default constructor initializes the enhanced learning system
     */
    EnhancedLearningSystem();
    
    /**
     * Parameterized constructor for legacy compatibility
     */
    EnhancedLearningSystem(int num_synapses, int num_neurons);
    
    /**
     * Destructor cleans up GPU memory and resources
     */
    ~EnhancedLearningSystem();
    
    // ====================================================================
    // INITIALIZATION AND CONFIGURATION
    // ====================================================================
    
    /**
     * Initialize the learning system with network parameters
     * @param num_neurons Total number of neurons in the network
     * @param num_synapses Total number of synapses in the network
     * @param num_modules Number of modular components
     * @return Success status of initialization
     */
    bool initialize(int num_neurons, int num_synapses, int num_modules);
    
    /**
     * Configure core learning parameters
     * @param lr Learning rate
     * @param decay Eligibility trace decay rate
     * @param scaling Reward signal scaling factor
     */
    void configure_learning_parameters(float lr, float decay, float scaling);
    
    /**
     * Setup modular architecture with variable module sizes
     * @param module_sizes Vector containing size of each module
     */
    void setup_modular_architecture(const std::vector<int>& module_sizes);
    
    /**
     * Configure which learning mechanisms are enabled
     * @param enable_stdp Enable spike-timing dependent plasticity
     * @param enable_homeostatic Enable homeostatic regulation
     * @param enable_correlation Enable correlation-based learning
     */
    void configure_learning_mechanisms(bool enable_stdp, bool enable_homeostatic, bool enable_correlation);
    
    /**
     * Set learning parameters for specific module
     * @param module_id ID of the target module
     * @param learning_rate Module-specific learning rate
     * @param plasticity_threshold Threshold for plasticity activation
     */
    void set_module_learning_parameters(int module_id, float learning_rate, float plasticity_threshold);
    
    /**
     * Configure reward prediction error learning
     * @param prediction_window Time window for prediction
     * @param error_sensitivity Sensitivity to prediction errors
     */
    void configure_reward_prediction(float prediction_window, float error_sensitivity);
    
    // ====================================================================
    // MAIN LEARNING UPDATE METHODS
    // ====================================================================
    
    /**
     * Main learning update coordinating all mechanisms
     * @param current_time Current simulation time
     * @param dt Time step
     * @param reward_signal External reward signal
     */
    void update_learning(float current_time, float dt, float reward_signal);
    
    /**
     * Update STDP and eligibility traces
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_stdp_and_eligibility(float current_time, float dt);
    
    /**
     * Apply reward modulation to synaptic weights
     * @param reward_signal Reward signal strength
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void apply_reward_modulation(float reward_signal, float current_time, float dt);
    
    /**
     * Update correlation-based learning mechanisms
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_correlation_learning(float current_time, float dt);
    
    /**
     * Apply homeostatic regulation to maintain network stability
     * @param target_activity Target activity level
     * @param dt Time step
     */
    void apply_homeostatic_regulation(float target_activity, float dt);
    
    /**
     * Update attention-modulated learning
     * @param attention_weights Attention weights for different modules
     * @param dt Time step
     */
    void update_attention_learning(const std::vector<float>& attention_weights, float dt);
    
    /**
     * Update learning for specific module
     * @param module_id Target module ID
     * @param module_reward Module-specific reward
     * @param dt Time step
     */
    void update_modular_learning(int module_id, float module_reward, float dt);
    
    /**
     * Set reward signal for learning system
     * @param reward_value Reward signal value
     */
    void setRewardSignal(float reward_value);
    
    /**
     * Trigger protein synthesis for long-term memory consolidation
     * @param stimulus_strength Strength of stimulus triggering synthesis
     */
    void triggerProteinSynthesis(float stimulus_strength);
    
    // ====================================================================
    // STATE MANAGEMENT
    // ====================================================================
    
    /**
     * Save current learning state to file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool save_learning_state(const std::string& filename) const;
    
    /**
     * Load learning state from file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool load_learning_state(const std::string& filename);
    
    /**
     * Reset all learning state to initial conditions
     */
    void reset_learning_state();
    
    // ====================================================================
    // PERFORMANCE MONITORING AND STATISTICS
    // ====================================================================
    
    /**
     * Learning system statistics structure
     */
    struct LearningStats {
        float total_weight_change;
        float average_trace_activity;
        float current_dopamine_level;
        float prediction_error;
        float network_activity;
        int plasticity_updates;
    };

    /**
     * Get average eligibility trace across network
     * @return Average eligibility trace value
     */
    float get_average_eligibility_trace() const;
    
    /**
     * Get overall learning progress metric
     * @return Learning progress value
     */
    float get_learning_progress() const;
    
    /**
     * Get learning rates for all modules
     * @return Vector of module learning rates
     */
    std::vector<float> get_module_learning_rates() const;
    
    /**
     * Get correlation statistics
     * @param stats Output vector for correlation statistics
     */
    void get_correlation_statistics(std::vector<float>& stats) const;
    
    /**
     * Get total weight change across network
     * @return Total magnitude of weight changes
     */
    float get_total_weight_change() const;
    
    /**
     * Get detailed learning statistics
     * @param detailed_stats Output vector for detailed statistics
     */
    void get_detailed_learning_statistics(std::vector<float>& detailed_stats) const;
    
    // ====================================================================
    // GPU-SPECIFIC LEARNING METHODS
    // ====================================================================
    
    /**
     * Update learning on GPU with direct GPU memory access
     * @param synapses GPU synapse array
     * @param neurons GPU neuron state array
     * @param current_time Current simulation time
     * @param dt Time step
     * @param external_reward External reward signal
     */
    void updateLearningGPU(struct GPUSynapse* synapses, 
                          struct GPUNeuronState* neurons,
                          float current_time, 
                          float dt,
                          float external_reward);
    
    /**
     * Reset episode on GPU
     * @param reset_traces Reset eligibility traces
     * @param reset_rewards Reset reward signals
     */
    void resetEpisodeGPU(bool reset_traces, bool reset_rewards);
    
    /**
     * Get learning statistics from GPU
     * @return Learning statistics structure
     */
    LearningStats getStatisticsGPU() const;

    // ====================================================================
    // LEGACY INTERFACE METHODS
    // ====================================================================
    
    /**
     * Main update function for legacy compatibility
     * This is called from the main network simulation loop
     */
    void updateLearning(GPUSynapse* synapses, 
                       GPUNeuronState* neurons,
                       float current_time, 
                       float dt,
                       float external_reward);
    
    /**
     * Get learning system statistics
     */
    LearningStats getStatistics() const;
    
    /**
     * Reset learning system state (for episodic learning)
     */
    void resetEpisode(bool reset_traces = true, bool reset_rewards = true);

private:
    // ====================================================================
    // PRIVATE IMPLEMENTATION METHODS
    // ====================================================================
    
    /**
     * Initialize CUDA resources and memory allocation
     * @return Success status of CUDA initialization
     */
    bool initialize_cuda_resources();
    
    /**
     * Clean up CUDA resources and free memory
     */
    void cleanup_cuda_resources();
    
    /**
     * Launch eligibility trace reset kernels
     */
    void launch_eligibility_reset();
    
    /**
     * Launch eligibility trace reset kernels on GPU
     */
    void launch_eligibility_reset_gpu();
    
    /**
     * Update performance metrics tracking
     */
    void update_performance_metrics();
    
    /**
     * Update performance metrics from GPU data
     */
    void update_performance_metrics_gpu();
    
    /**
     * Reset eligibility traces on GPU
     */
    void reset_eligibility_traces_gpu();
};

#endif // ENHANCED_LEARNING_SYSTEM_H