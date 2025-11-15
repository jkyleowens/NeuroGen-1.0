#ifndef NEURAL_MODULE_H
#define NEURAL_MODULE_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <map>
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>

/**
 * @brief Base Neural Module for Modular Neural Network Architecture
 * 
 * Self-contained neural network module with independent state management,
 * CUDA integration, and inter-module communication capabilities.
 * Designed for biological brain-like modular neural networks.
 */
class NeuralModule {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct neural module with configuration
     * @param name Module name for identification
     * @param config Network configuration parameters
     */
    NeuralModule(const std::string& name, const NetworkConfig& config);
    
    /**
     * @brief Virtual destructor for polymorphic inheritance
     */
    virtual ~NeuralModule();
    
    /**
     * @brief Initialize the neural module with biological parameters
     * @return Success status of initialization
     */
    virtual bool initialize();
    
    /**
     * @brief Update module with time step, inputs, and reward signal
     * @param dt Time step in seconds
     * @param inputs Input vector to process (optional)
     * @param reward Reward signal for learning (optional)
     */
    virtual void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f);
    
    /**
     * @brief Validate module configuration and state
     * @return Validation success status
     */
    bool validate_configuration() const;
    
    // ========================================================================
    // CORE PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process input through the neural module
     * @param input Input vector to process
     * @return Processed output vector
     */
    virtual std::vector<float> process(const std::vector<float>& input);
    
    /**
     * @brief Get current module output
     * @return Current output activation values
     */
    virtual std::vector<float> get_output() const;
    
    /**
     * @brief Get specific neuron potentials
     * @param neuron_ids Vector of neuron IDs to query
     * @return Vector of membrane potentials
     */
    std::vector<float> get_neuron_potentials(const std::vector<size_t>& neuron_ids) const;
    
    // ========================================================================
    // MODULE STATE AND CONTROL
    // ========================================================================
    
    /**
     * @brief Set module active state
     * @param active Activity state to set
     */
    virtual void set_active(bool active);
    
    /**
     * @brief Check if module is active
     * @return Current activity state
     */
    virtual bool is_active() const;
    
    /**
     * @brief Get module name
     * @return Module identifier string
     */
    const std::string& get_name() const;
    
    /**
     * @brief Get internal network reference
     * @return Pointer to internal network
     */
    Network* get_network();
    
    /**
     * @brief Get module statistics
     * @return Current network statistics
     */
    NetworkStats get_stats() const;
    
    // ========================================================================
    // INTER-MODULE COMMUNICATION
    // ========================================================================
    
    /**
     * @brief Register neuron port for inter-module connections
     * @param port_name Name of the port
     * @param neuron_ids Vector of neuron IDs in this port
     */
    void register_neuron_port(const std::string& port_name, const std::vector<size_t>& neuron_ids);
    
    /**
     * @brief Get neuron population for a port
     * @param port_name Name of the port
     * @return Vector of neuron IDs
     */
    const std::vector<size_t>& get_neuron_population(const std::string& port_name) const;
    
    /**
     * @brief Send signal to another module
     * @param signal Vector of signal values
     * @param target_module Target module name
     * @param target_port Target port name
     */
    virtual void send_signal(const std::vector<float>& signal, 
                           const std::string& target_module,
                           const std::string& target_port);
    
    /**
     * @brief Receive signal from another module
     * @param signal Vector of signal values
     * @param source_module Source module name
     * @param source_port Source port name
     */
    virtual void receive_signal(const std::vector<float>& signal,
                              const std::string& source_module,
                              const std::string& source_port);
    
    /**
     * @brief Apply neuromodulation to the module
     * @param modulator_type Type of neuromodulator (dopamine, serotonin, etc.)
     * @param level Modulation level
     */
    virtual void applyNeuromodulation(const std::string& modulator_type, float level);
    
    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get performance metrics
     * @return Map of performance metric names to values
     */
    virtual std::map<std::string, float> getPerformanceMetrics() const;
    
    /**
     * @brief Reset performance counters
     */
    void reset_performance_metrics();
    
    /**
     * @brief Update performance tracking
     * @param dt Time step for this update
     */
    void update_performance_metrics(float dt);
    
    // ========================================================================
    // STATE SERIALIZATION
    // ========================================================================
    
    /**
     * @brief Save module state to file
     * @param filename Output filename
     * @return Success status
     */
    virtual bool save_state(const std::string& filename) const;
    
    /**
     * @brief Load module state from file
     * @param filename Input filename
     * @return Success status
     */
    virtual bool load_state(const std::string& filename);
    
    // ========================================================================
    // CUDA INTEGRATION
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources for this module
     * @return Success status
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Clean up CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Check if CUDA is available and initialized
     * @return CUDA availability status
     */
    bool is_cuda_available() const;

protected:
    // Core module properties
    std::string module_name_;
    NetworkConfig config_;
    
    // Neural network instance
    std::unique_ptr<Network> internal_network_;
    
    // Module state
    bool active_;
    bool is_initialized_;
    bool cuda_initialized_;
    
    // Inter-module communication
    std::unordered_map<std::string, std::vector<size_t>> neuron_ports_;
    std::vector<float> outgoing_signals_;
    std::vector<float> incoming_signals_;
    
    // Neural state management
    std::vector<float> internal_state_;
    std::vector<float> activation_history_;
    std::vector<float> synaptic_weights_;
    std::vector<float> neuron_outputs_;

    // Configuration
    std::string module_id_;
    std::string module_type_;
    size_t num_neurons_;
    size_t num_synapses_;
    float learning_rate_;
    float plasticity_strength_;
    float homeostatic_target_;

    // State
    float average_activity_;
    size_t update_count_;

    // Thread safety
    mutable std::mutex module_mutex_;

private:
    // Learning and plasticity
    bool plasticity_enabled_;
    
    // Biological parameters
    float excitability_level_;
    float adaptation_current_;
    float background_noise_;
    float refractory_period_;
    
    // Performance metrics
    float firing_rate_;
    float connection_strength_;
    float plasticity_events_;
    float last_update_time_;
    
    // Internal methods
    void calculateAverageActivity();
    
protected:
    // ========================================================================
    // INTERNAL PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Apply nonlinear activation function
     * @param input Raw input value
     * @return Activated output
     */
    virtual float apply_activation(float input) const;
    
    /**
     * @brief Update synaptic weights using STDP
     * @param pre_activity Presynaptic activity
     * @param post_activity Postsynaptic activity
     * @param dt Time step
     */
    virtual void update_synaptic_weights(const std::vector<float>& pre_activity,
                                       const std::vector<float>& post_activity, float dt);
    
    /**
     * @brief Initialize synaptic weights with biological distribution
     */
    virtual void initialize_synaptic_weights();
    
    /**
     * @brief Update activity history for temporal processing
     * @param current_activity Current activity level
     */
    virtual void update_activity_history(float current_activity);
    
    /**
     * @brief Compute firing rate from activity history
     * @return Computed firing rate
     */
    virtual float compute_firing_rate() const;
    
    /**
     * @brief Apply noise for biological realism
     * @param signal Input signal
     * @return Noisy signal
     */
    virtual float apply_biological_noise(float signal) const;
    
    /**
     * @brief Update internal performance counters
     */
    void update_internal_metrics();
};

#endif // NEURAL_MODULE_H