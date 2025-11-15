// ============================================================================
// SPECIALIZED MODULE CLASS - Cortical Processing Units
// File: include/NeuroGen/SpecializedModule.h
// ============================================================================

#ifndef SPECIALIZED_MODULE_H
#define SPECIALIZED_MODULE_H

#include <vector>
#include <string>
#include <memory>
#include <NeuroGen/EnhancedNeuralModule.h>

/**
 * @brief Specialized Module for Cortical Processing
 * 
 * Implements specialized neural processing units that mimic different cortical areas:
 * - Motor cortex for movement planning and execution
 * - Attention system for resource allocation and focus control
 * - Reward system for value estimation and learning
 * - Working memory for temporary information maintenance
 * 
 * Each specialized module has its own internal state, processing characteristics,
 * and biological parameters to simulate realistic cortical function.
 */
class SpecializedModule : public EnhancedNeuralModule {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct specialized module with configuration
     * @param name Module name for identification
     * @param config Network configuration parameters
     * @param module_type Type of specialization (motor, attention, reward, memory)
     */
    SpecializedModule(const std::string& name, const NetworkConfig& config, 
                     const std::string& module_type = "general");
    
    /**
     * @brief Virtual destructor
     */
    virtual ~SpecializedModule() = default;
    
    /**
     * @brief Initialize the specialized module
     * @return Success status of initialization
     */
    bool initialize() override;
    
    // ========================================================================
    // SPECIALIZED PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process motor cortex computations
     * @param motor_input Vector of motor command inputs
     * @return Vector of processed motor outputs
     */
    std::vector<float> process_motor_cortex(const std::vector<float>& motor_input);
    
    /**
     * @brief Process attention system computations
     * @param attention_input Vector of attention control inputs
     * @return Vector of processed attention outputs
     */
    std::vector<float> process_attention_system(const std::vector<float>& attention_input);
    
    /**
     * @brief Process reward system computations
     * @param reward_input Vector of reward signal inputs
     * @return Vector of processed reward prediction outputs
     */
    std::vector<float> process_reward_system(const std::vector<float>& reward_input);
    
    /**
     * @brief Process working memory computations
     * @param memory_input Vector of memory content inputs
     * @return Vector of processed memory outputs
     */
    std::vector<float> process_working_memory(const std::vector<float>& memory_input);
    
    // ========================================================================
    // MODULE CONFIGURATION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Set module specialization type
     * @param type Specialization type string
     */
    void set_specialization_type(const std::string& type);
    
    /**
     * @brief Get module specialization type
     * @return Current specialization type
     */
    const std::string& get_specialization_type() const;
    
    /**
     * @brief Set attention weight for this module
     * @param weight New attention weight value
     */
    void set_attention_weight(float weight);

    /**
     * @brief Applies reinforcement signal to the module.
     * @param reward The reward signal.
     * @param global_reward The global reward signal.
     */
    void apply_reinforcement(float reward, float global_reward);
    
    // ========================================================================
    // GETTERS AND SETTERS
    // ========================================================================
    
    /**
     * @brief Get current attention weight
     * @return Current attention weight
     */
    float get_attention_weight() const;
    
    /**
     * @brief Set activation threshold
     * @param threshold Activation threshold value
     */
    void set_activation_threshold(float threshold);
    
    /**
     * @brief Get current activation threshold
     * @return Current activation threshold
     */
    float get_activation_threshold() const;
    
    // ========================================================================
    // OVERRIDDEN VIRTUAL FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Update module with specialized processing
     * @param dt Time step in seconds
     * @param inputs Input vector to process (optional)
     * @param reward Reward signal (optional)
     */
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    /**
     * @brief Get module output
     * @return Current output vector
     */
    std::vector<float> get_output() const override;

protected:
    // ========================================================================
    // INTERNAL STATE AND PARAMETERS
    // ========================================================================
    
    /** Module specialization type */
    std::string specialization_type_;
    
    /** Internal processing state */
    std::vector<float> internal_state_;
    
    /** Output buffer for processed results */
    std::vector<float> output_buffer_;
    
    /** Attention weight for modulating processing */
    float attention_weight_;
    
    /** Activation threshold for neural responses */
    float activation_threshold_;
    
    /** Learning rate for internal adaptation */
    float learning_rate_;
    
    /** Decay rate for internal state */
    float decay_rate_;
    
    /** Noise level for biological realism */
    float noise_level_;
    
    // ========================================================================
    // HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize internal state and buffers
     * @param state_size Size of internal state vector
     * @param output_size Size of output buffer
     */
    void initialize_internal_state(size_t state_size, size_t output_size);
    
    /**
     * @brief Apply biological noise to signal
     * @param signal Input signal
     * @return Noisy signal
     */
    float apply_noise(float signal) const;
    
    /**
     * @brief Update internal metrics and statistics
     */
    void update_internal_metrics();
};

#endif // SPECIALIZED_MODULE_H
