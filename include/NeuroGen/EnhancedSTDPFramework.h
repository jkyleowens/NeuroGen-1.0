#ifndef ENHANCED_STDP_FRAMEWORK_H
#define ENHANCED_STDP_FRAMEWORK_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <NeuroGen/NeuralConstants.h>

// Forward declarations (no CUDA dependencies)
struct GPUSynapse;
struct GPUNeuronState;
struct GPUPlasticityState;
struct GPUNeuromodulatorState;

/**
 * @brief Enhanced STDP Framework with Comprehensive Biological Realism
 * 
 * This framework implements multiple forms of synaptic plasticity that mirror
 * the sophistication found in biological neural circuits, including:
 * - Classical STDP with precise timing dependence
 * - BCM learning rule with sliding threshold
 * - Homeostatic synaptic scaling
 * - Neuromodulatory influences (dopamine, acetylcholine, serotonin)
 * - Meta-plasticity mechanisms
 * - Multi-compartment dendritic integration
 */
class EnhancedSTDPFramework {
private:
    // GPU memory management (using void* to avoid CUDA types)
    void* d_synapses_;
    void* d_neurons_;
    void* d_plasticity_states_;
    void* d_neuromodulator_states_;
    void* d_stdp_config_;
    
    // Network dimensions
    int num_synapses_;
    int num_neurons_;
    bool cuda_initialized_;
    
    // Learning parameters
    float stdp_learning_rate_;
    float bcm_learning_rate_;
    float homeostatic_rate_;
    float neuromodulation_strength_;
    float metaplasticity_rate_;
    
    // Performance tracking
    mutable std::mutex framework_mutex_;
    mutable float total_weight_change_;
    mutable float plasticity_events_;
    mutable float last_update_time_;
    mutable float average_eligibility_trace_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    EnhancedSTDPFramework();
    ~EnhancedSTDPFramework();
    
    /**
     * @brief Initialize the STDP framework with network specifications
     * @param num_neurons Total number of neurons in the network
     * @param num_synapses Total number of synapses in the network
     * @return Success status of initialization
     */
    bool initialize(int num_neurons, int num_synapses);
    
    /**
     * @brief Configure comprehensive learning parameters
     * @param stdp_rate STDP learning rate for spike-timing dependent plasticity
     * @param bcm_rate BCM learning rate for threshold adaptation
     * @param homeostatic_rate Rate of homeostatic regulation
     * @param neuromod_strength Strength of neuromodulatory influences
     */
    void configure_learning_parameters(float stdp_rate, float bcm_rate, 
                                     float homeostatic_rate, float neuromod_strength);
    
    // ========================================================================
    // MAIN PLASTICITY MECHANISMS - C++ WRAPPER INTERFACE
    // ========================================================================
    
    /**
     * @brief Execute enhanced STDP with multi-factor modulation
     * @param current_time Current simulation time (milliseconds)
     * @param dt Integration time step (milliseconds)
     */
    void update_enhanced_stdp(float current_time, float dt);
    
    /**
     * @brief Apply BCM learning rule with adaptive threshold
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_bcm_learning(float current_time, float dt);
    
    /**
     * @brief Execute homeostatic regulation mechanisms
     * @param target_activity Target network activity level (Hz)
     * @param dt Time step
     */
    void update_homeostatic_regulation(float target_activity, float dt);
    
    /**
     * @brief Apply neuromodulatory influences on plasticity
     * @param dopamine_level Global dopamine concentration
     * @param acetylcholine_level Global acetylcholine level
     * @param dt Time step
     */
    void update_neuromodulation(float dopamine_level, float acetylcholine_level, float dt);
    
    /**
     * @brief Update meta-plasticity mechanisms
     * @param experience_level Current experience level
     * @param dt Time step
     */
    void update_metaplasticity(float experience_level, float dt);
    
    /**
     * @brief Execute comprehensive plasticity update (all mechanisms)
     * @param current_time Current simulation time
     * @param dt Time step
     * @param dopamine_level Dopamine concentration
     * @param target_activity Target activity level
     */
    void update_all_plasticity_mechanisms(float current_time, float dt, 
                                        float dopamine_level, float target_activity);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get total synaptic weight change magnitude
     * @return Cumulative weight change since last reset
     */
    float get_total_weight_change() const;
    
    /**
     * @brief Get number of plasticity events
     * @return Count of significant plasticity events
     */
    float get_plasticity_events() const;
    
    /**
     * @brief Get average synaptic weight across network
     * @return Mean synaptic weight
     */
    float get_average_synaptic_weight() const;
    
    /**
     * @brief Get average eligibility trace magnitude
     * @return Mean eligibility trace value
     */
    float get_average_eligibility_trace() const;
    
    /**
     * @brief Get comprehensive plasticity statistics
     * @param stats Output vector containing detailed statistics
     */
    void get_plasticity_statistics(std::vector<float>& stats) const;
    
    /**
     * @brief Generate detailed plasticity report
     * @param filename Output filename for detailed report
     */
    void generate_plasticity_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save complete plasticity state to file
     * @param filename Base filename for state persistence
     * @return Success status
     */
    bool save_plasticity_state(const std::string& filename) const;
    
    /**
     * @brief Load complete plasticity state from file
     * @param filename Base filename for state loading
     * @return Success status
     */
    bool load_plasticity_state(const std::string& filename);
    
    /**
     * @brief Reset all plasticity mechanisms to baseline
     */
    void reset_plasticity_state();
    
    /**
     * @brief Set plasticity mechanism enable/disable flags
     * @param enable_stdp Enable STDP mechanisms
     * @param enable_bcm Enable BCM learning
     * @param enable_homeostatic Enable homeostatic regulation
     * @param enable_neuromodulation Enable neuromodulatory influences
     */
    void configure_plasticity_mechanisms(bool enable_stdp, bool enable_bcm, 
                                       bool enable_homeostatic, bool enable_neuromodulation);

private:
    // ========================================================================
    // INTERNAL CUDA WRAPPER FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources and memory allocation
     * @return Success status of CUDA initialization
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Release all CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Launch enhanced STDP kernel (internal wrapper)
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void launch_enhanced_stdp_kernel(float current_time, float dt);
    
    /**
     * @brief Launch BCM learning kernel (internal wrapper)
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void launch_bcm_learning_kernel(float current_time, float dt);
    
    /**
     * @brief Launch homeostatic regulation kernel (internal wrapper)
     * @param target_activity Target activity level
     * @param dt Time step
     */
    void launch_homeostatic_kernel(float target_activity, float dt);
    
    /**
     * @brief Launch neuromodulation kernel (internal wrapper)
     * @param dopamine_level Dopamine concentration
     * @param acetylcholine_level Acetylcholine level
     * @param dt Time step
     */
    void launch_neuromodulation_kernel(float dopamine_level, float acetylcholine_level, float dt);
    
    /**
     * @brief Update performance metrics from GPU
     */
    void update_performance_metrics();
    
    /**
     * @brief Validate CUDA operation success
     * @param operation_name Operation name for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
    
    /**
     * @brief Copy statistics from GPU to CPU
     */
    void copy_statistics_from_gpu();
    
    /**
     * @brief Configure optimal GPU execution parameters
     */
    void configure_gpu_execution_parameters();
};

#endif // ENHANCED_STDP_FRAMEWORK_H