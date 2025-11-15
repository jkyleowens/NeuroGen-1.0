#ifndef ENHANCED_NEURAL_MODULE_H
#define ENHANCED_NEURAL_MODULE_H

#include <NeuroGen/NeuralModule.h>
#include <NeuroGen/TopologyGenerator.h>
#include <NeuroGen/LearningState.h>
#include <memory>
#include <functional>
#include <chrono>
#include <queue>
#include <map>

/**
 * Enhanced neural module with biological features:
 * - Central control/attention mechanism
 * - Internal feedback loops
 * - State saving/loading per module
 * - Inter-module communication pathways
 */
class EnhancedNeuralModule : public NeuralModule {
public:
    // Module state for serialization
    struct ModuleState {
        std::string module_name;
        std::vector<float> neuron_states;
        std::vector<float> synapse_weights;
        std::vector<float> neuromodulator_levels;
        float module_attention_weight;
        int developmental_stage;
        std::map<std::string, float> performance_metrics;
        std::chrono::steady_clock::time_point timestamp;
        bool is_active;
    };
    
    // Inter-module connection specification
    struct InterModuleConnection {
        std::string source_port;
        std::string target_module;
        std::string target_port;
        float connection_strength;
        bool is_feedback;  // True for feedback connections
        float delay_ms;    // Transmission delay
    };

    // Constructor
    EnhancedNeuralModule(const std::string& name, const NetworkConfig& config)
        : NeuralModule(name, config), 
          attention_weight_(1.0f),
          is_active_(true),
          developmental_stage_(0),
          last_feedback_update_(std::chrono::steady_clock::now()) {}

    // Virtual destructor
    virtual ~EnhancedNeuralModule() = default;

    // ========================================================================
    // OVERRIDDEN VIRTUAL FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize the enhanced neural module
     * @return Success status of initialization
     */
    bool initialize() override;
    
    /**
     * @brief Update module with enhanced biological features
     * @param dt Time step in seconds
     * @param inputs Input vector to process (optional)
     * @param reward Reward signal for learning (optional)
     */
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    /**
     * @brief Get enhanced performance metrics including biological features
     * @return Map of performance metric names to values
     */
    std::map<std::string, float> getPerformanceMetrics() const override;
    
    /**
     * @brief Get current neural outputs from the module
     * @return Vector of current neuron outputs
     */
    std::vector<float> getOutputs() const;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save complete module state including biological parameters
     * @return ModuleState structure with all state information
     */
    virtual ModuleState saveState() const;
    
    /**
     * @brief Load complete module state
     * @param state ModuleState structure to load
     */
    virtual void loadState(const ModuleState& state);

    /**
     * @brief Get the current activity level of the module
     * @return Activity level (e.g., based on attention)
     */
    virtual float getActivityLevel() const;
    
    // ========================================================================
    // ATTENTION AND MODULATION
    // ========================================================================
    
    /**
     * @brief Set attention weight for this module
     * @param weight Attention weight [0-1]
     */
    void setAttentionWeight(float weight) { attention_weight_ = weight; }
    
    /**
     * @brief Get current attention weight
     * @return Current attention weight
     */
    float getAttentionWeight() const { return attention_weight_; }
    
    /**
     * @brief Update attention based on context
     * @param context_vector Context information
     */
    void updateAttention(const std::vector<float>& context_vector);
    
    // ========================================================================
    // BIOLOGICAL FEATURES
    // ========================================================================
    
    /**
     * @brief Set developmental stage of the module
     * @param stage Developmental stage (0-5)
     */
    void setDevelopmentalStage(int stage) { developmental_stage_ = stage; }
    
    /**
     * @brief Get current developmental stage
     * @return Developmental stage
     */
    int getDevelopmentalStage() const { return developmental_stage_; }
    
    /**
     * @brief Apply neuromodulation to the module
     * @param modulator_type Type of neuromodulator
     * @param level Modulation level
     */
    virtual void applyNeuromodulation(const std::string& modulator_type, float level) override;
    
    /**
     * @brief Process internal feedback loops
     * @param dt Time step
     */
    void processFeedbackLoops(float dt);
    
    /**
     * @brief Add inter-module connection
     * @param connection Connection specification
     */
    void addInterModuleConnection(const InterModuleConnection& connection);
    
    /**
     * @brief Register inter-module connection (alias for addInterModuleConnection)
     * @param connection Connection specification
     */
    void registerInterModuleConnection(const InterModuleConnection& connection) {
        addInterModuleConnection(connection);
    }
    
    /**
     * @brief Send signal to connected modules
     * @param signal_data Data to send
     * @param target_port Target port name
     */
    void sendInterModuleSignal(const std::vector<float>& signal_data, 
                               const std::string& target_port);
    
    /**
     * @brief Receive signal from other modules
     * @param signal_data Received data
     * @param source_port Source port name
     */
    void receiveInterModuleSignal(const std::vector<float>& signal_data,
                                  const std::string& source_port);
    
    // =======================================================================
    // LEARNING AND PLASTICITY
    // =======================================================================

    /**
     * @brief Get the current learning state of the module
     * @return ModuleLearningState structure with learning-related data
     */
    ModuleLearningState getLearningState() const;

    /**
     * @brief Apply a learning state to the module
     * @param state The learning state to apply
     * @return True if the state was applied successfully, false otherwise
     */
    bool applyLearningState(const ModuleLearningState& state);

    /**
     * @brief Get the synaptic weights of the module
     * @return A vector of synaptic weights
     */
    std::vector<float> getSynapticWeights() const;

    /**
     * @brief Set the synaptic weights of the module
     * @param weights A vector of synaptic weights
     * @return True if the weights were set successfully, false otherwise
     */
    bool setSynapticWeights(const std::vector<float>& weights);

    /**
     * @brief Perform memory consolidation based on tagged synapses
     * @param consolidation_strength The strength of the consolidation
     * @return The number of synapses consolidated
     */
    size_t performMemoryConsolidation(float consolidation_strength);

    /**
     * @brief Update neuromodulator levels
     * @param dopamine Dopamine level
     * @param acetylcholine Acetylcholine level
     * @param norepinephrine Norepinephrine level
     */
    void updateNeuromodulators(float dopamine, float acetylcholine, float norepinephrine);
    
    // =======================================================================
    // ACTIVITY AND STATUS
    // ========================================================================
    
    /**
     * @brief Check if module is currently active
     * @return Activity status
     */
    bool isActive() const { return is_active_; }
    
    /**
     * @brief Set module activity status
     * @param active New activity status
     */
    void setActive(bool active) { is_active_ = active; }
    
    /**
     * @brief Get module specialization type
     * @return Specialization description
     */
    virtual std::string getSpecializationType() const { return "general"; }

protected:
    // =======================================================================
    // PROTECTED MEMBER VARIABLES
    // =======================================================================
    
    float attention_weight_;
    bool is_active_;
    int developmental_stage_;
    std::chrono::steady_clock::time_point last_feedback_update_;
    
    // Inter-module communication
    std::vector<InterModuleConnection> connections_;
    std::map<std::string, std::queue<std::vector<float>>> input_buffers_;
    std::map<std::string, std::vector<float>> output_buffers_;
    
    // Feedback loop state
    std::vector<float> feedback_state_;
    float feedback_strength_;
    
    // Neuromodulator levels
    std::map<std::string, float> neuromodulator_levels_;
    
    // Learning and Plasticity
    std::vector<float> eligibility_traces_;
    std::vector<float> synaptic_tags_;
    std::vector<float> firing_rate_buffer_;
    std::vector<float> prediction_error_history_;
    size_t performance_history_index_;
    
    float dopamine_level_ = 0.0f;
    float acetylcholine_level_ = 0.0f;
    float norepinephrine_level_ = 0.0f;
    
    float synaptic_tag_threshold_ = 0.5f;
    float eligibility_decay_rate_ = 0.95f;
    float consolidation_threshold_ = 1.0f;

    std::chrono::steady_clock::time_point last_consolidation_;
    bool consolidation_pending_ = false;

    // =======================================================================
    // PROTECTED HELPER METHODS
    // =======================================================================
    
    /**
     * @brief Update internal biological processes
     * @param dt Time step
     */
    void updateBiologicalProcesses(float dt);
    
    /**
     * @brief Initialize learning traces and buffers
     */
    void initializeLearningTraces();

    /**
     * @brief Update eligibility traces for synapses
     * @param reward_signal The global reward signal
     * @param dt Time step
     */
    void updateEligibilityTraces(float reward_signal, float dt);

    /**
     * @brief Apply synaptic tags based on novelty
     * @param novelty_signal A signal indicating novelty
     */
    void applySynapticTagging(float novelty_signal);

    /**
     * @brief Calculate prediction error
     * @param expected_output The expected output vector
     * @return The calculated prediction error
     */
    float getPredictionError(const std::vector<float>& expected_output) const;

    /**
     * @brief Update the history of performance metrics
     * @param prediction_error The latest prediction error
     */
    void updatePerformanceHistory(float prediction_error);

    /**
     * @brief Check if memory consolidation should be performed
     * @return True if consolidation is needed, false otherwise
     */
    bool shouldConsolidate() const;

    /**
     * @brief Compute attention-weighted output
     * @param raw_output Raw module output
     * @return Attention-weighted output
     */
    std::vector<float> applyAttentionWeighting(const std::vector<float>& raw_output) const;
    
    /**
     * @brief Process developmental changes
     * @param dt Time step
     */
    void updateDevelopmentalState(float dt);
    
    /**
     * @brief Process inter-module communication signals
     */
    void processInterModuleCommunication();
};

#endif // ENHANCED_NEURAL_MODULE_H