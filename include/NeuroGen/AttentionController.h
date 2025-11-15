// ============================================================================
// ATTENTION CONTROLLER HEADER - NLP-FOCUSED ARCHITECTURE
// File: include/NeuroGen/AttentionController.h
// ============================================================================

#ifndef ATTENTION_CONTROLLER_H
#define ATTENTION_CONTROLLER_H

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <functional>

/**
 * @brief Attention Controller for Neural Module Coordination
 * 
 * This class manages attention allocation across neural modules in the
 * brain-inspired architecture. It implements biologically-plausible
 * attention mechanisms including:
 * 
 * - Context-dependent attention allocation
 * - Competitive inhibition between modules
 * - Attention decay and plasticity
 * - Task-specific attention weighting
 * - Global attention coordination
 * 
 * Optimized for NLP processing with specific support for:
 * - Central Controller (neuromodulatory control)
 * - Input Module (text processing)
 * - Language Processing Module
 * - Reasoning Module
 * - Output Module
 */
class AttentionController {
public:
    // ========================================================================
    // ATTENTION CONFIGURATION
    // ========================================================================
    
    struct AttentionConfig {
        float attention_decay_rate = 0.95f;        // Rate of attention decay over time
        float attention_boost_threshold = 0.7f;    // Threshold for attention boosting
        float global_inhibition_strength = 0.1f;   // Strength of inter-module competition
        float plasticity_rate = 0.01f;             // Rate of attention weight adaptation
        float context_sensitivity = 0.5f;          // Sensitivity to context changes
        float language_attention_bias = 1.2f;      // Bias towards language processing
        bool enable_competitive_inhibition = true;  // Enable competition between modules
        bool enable_attention_plasticity = true;    // Enable adaptive attention weights
    };
    
    struct ModuleAttentionState {
        std::string module_name;
        float current_weight = 1.0f;
        float baseline_weight = 1.0f;
        float activation_history = 0.0f;
        float performance_score = 0.5f;
        std::chrono::steady_clock::time_point last_update;
        bool is_active = true;
        std::vector<float> attention_history;
    };
    
    enum class AttentionMode {
        BALANCED,           // Equal attention to all modules
        LANGUAGE_FOCUSED,   // Focus on language processing
        REASONING_FOCUSED,  // Focus on reasoning module
        INPUT_FOCUSED,      // Focus on input processing
        OUTPUT_FOCUSED,     // Focus on output generation
        ADAPTIVE           // Dynamically adjust based on context
    };
    
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct attention controller with default configuration
     */
    AttentionController();
    
    /**
     * @brief Construct attention controller with custom configuration
     * @param config Attention configuration parameters
     */
    explicit AttentionController(const AttentionConfig& config);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~AttentionController() = default;
    
    /**
     * @brief Initialize attention controller
     * @return Success status
     */
    bool initialize();
    
    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();
    
    // ========================================================================
    // MODULE REGISTRATION AND MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Register a neural module for attention control
     * @param module_name Name of the module to register
     * @param baseline_weight Initial attention weight (default: 1.0)
     * @return Success status
     */
    bool register_module(const std::string& module_name, float baseline_weight = 1.0f);
    
    /**
     * @brief Unregister a neural module
     * @param module_name Name of the module to unregister
     * @return Success status
     */
    bool unregister_module(const std::string& module_name);
    
    /**
     * @brief Check if module is registered
     * @param module_name Module name to check
     * @return True if module is registered
     */
    bool is_module_registered(const std::string& module_name) const;
    
    /**
     * @brief Get list of all registered modules
     * @return Vector of module names
     */
    std::vector<std::string> get_registered_modules() const;
    
    /**
     * @brief Get number of registered modules
     * @return Module count
     */
    size_t get_module_count() const;
    
    // ========================================================================
    // ATTENTION WEIGHT MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Get current attention weight for a module
     * @param module_name Name of the module
     * @return Attention weight (0.0 to 2.0, typically around 1.0)
     */
    float get_attention_weight(const std::string& module_name) const;
    
    /**
     * @brief Set attention weight for a module
     * @param module_name Name of the module
     * @param weight New attention weight
     * @return Success status
     */
    bool set_attention_weight(const std::string& module_name, float weight);
    
    /**
     * @brief Get attention weights for all modules
     * @return Vector of attention weights in registration order
     */
    std::vector<float> get_all_attention_weights() const;
    
    /**
     * @brief Get attention weights as map
     * @return Map of module names to attention weights
     */
    std::map<std::string, float> get_attention_weight_map() const;
    
    /**
     * @brief Reset all attention weights to baseline
     */
    void reset_attention_weights();
    
    /**
     * @brief Normalize attention weights to sum to module count
     */
    void normalize_attention_weights();
    
    // ========================================================================
    // CONTEXT-BASED ATTENTION CONTROL
    // ========================================================================
    
    /**
     * @brief Update attention based on current context
     * @param context_vector Context representation vector
     */
    void update_context(const std::vector<float>& context_vector);
    
    /**
     * @brief Update attention for language processing context
     * @param text_complexity Complexity of current text input
     * @param reasoning_demand Reasoning complexity required
     * @param response_urgency Urgency of response generation
     */
    void update_language_context(float text_complexity, 
                                float reasoning_demand, 
                                float response_urgency);
    
    /**
     * @brief Set task-specific attention priorities
     * @param task_type Type of current task
     * @param priority_map Map of module names to priority values
     */
    void set_task_priorities(const std::string& task_type, 
                            const std::map<std::string, float>& priority_map);
    
    /**
     * @brief Set context priority for specific situations
     * @param context Context identifier
     * @param priority Priority value (0.0 to 2.0)
     */
    void set_context_priority(const std::string& context, float priority);

    // Compatibility wrapper for legacy calls
    inline void set_priority(const std::string& context, float priority) {
        set_context_priority(context, priority);
    }
    
    /**
     * @brief Get current context priorities
     * @return Map of context names to priorities
     */
    std::map<std::string, float> get_context_priorities() const;
    
    // ========================================================================
    // ATTENTION MODES AND CONTROL
    // ========================================================================
    
    /**
     * @brief Set attention mode
     * @param mode Attention allocation mode
     */
    void set_attention_mode(AttentionMode mode);
    
    /**
     * @brief Get current attention mode
     * @return Current attention mode
     */
    AttentionMode get_attention_mode() const;
    
    /**
     * @brief Apply global inhibition across all modules
     * @param strength Inhibition strength (0.0 to 1.0)
     */
    void apply_global_inhibition(float strength);
    
    /**
     * @brief Enable or disable competitive inhibition
     * @param enable Whether to enable competitive inhibition
     */
    void enable_competitive_inhibition(bool enable);
    
    /**
     * @brief Boost attention for specific module temporarily
     * @param module_name Module to boost
     * @param boost_amount Amount of boost (multiplicative)
     * @param duration_ms Duration of boost in milliseconds
     */
    void boost_module_attention(const std::string& module_name, 
                               float boost_amount, 
                               int duration_ms = 1000);
    
    // ========================================================================
    // DYNAMIC ATTENTION COMPUTATION
    // ========================================================================
    
    /**
     * @brief Compute attention weights based on current state
     */
    void compute_attention_weights();
    
    /**
     * @brief Update attention weights with temporal dynamics
     * @param dt Time step in seconds
     */
    void update_attention_dynamics(float dt);
    
    /**
     * @brief Apply attention plasticity based on module performance
     * @param module_performance Map of module names to performance scores
     */
    void apply_attention_plasticity(const std::map<std::string, float>& module_performance);
    
    /**
     * @brief Update module performance history
     * @param module_name Module name
     * @param performance_score Performance score (0.0 to 1.0)
     */
    void update_module_performance(const std::string& module_name, float performance_score);
    
    // ========================================================================
    // NLP-SPECIFIC ATTENTION METHODS
    // ========================================================================
    
    /**
     * @brief Configure attention for NLP processing pipeline
     */
    void configure_for_nlp();
    
    /**
     * @brief Set language processing attention weights
     * @param input_weight Weight for input module
     * @param language_weight Weight for language processing
     * @param reasoning_weight Weight for reasoning module
     * @param output_weight Weight for output module
     */
    void set_nlp_attention_weights(float input_weight, 
                                  float language_weight, 
                                  float reasoning_weight, 
                                  float output_weight);
    
    /**
     * @brief Adapt attention for different language tasks
     * @param task_type Type of language task (e.g., "comprehension", "generation", "reasoning")
     */
    void adapt_for_language_task(const std::string& task_type);
    
    /**
     * @brief Focus attention on language processing modules
     * @param focus_strength Strength of language focus (0.0 to 2.0)
     */
    void focus_on_language_processing(float focus_strength = 1.5f);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get detailed attention state for a module
     * @param module_name Module name
     * @return Module attention state
     */
    ModuleAttentionState get_module_attention_state(const std::string& module_name) const;
    
    /**
     * @brief Get attention statistics
     * @return Map of statistic names to values
     */
    std::map<std::string, float> get_attention_statistics() const;
    
    /**
     * @brief Get attention entropy (measure of attention distribution)
     * @return Entropy value
     */
    float get_attention_entropy() const;
    
    /**
     * @brief Get attention focus (measure of attention concentration)
     * @return Focus value (0.0 to 1.0)
     */
    float get_attention_focus() const;
    
    /**
     * @brief Check if attention is stable
     * @return True if attention weights are stable
     */
    bool is_attention_stable() const;
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Get current configuration
     * @return Attention configuration
     */
    AttentionConfig get_config() const;
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void set_config(const AttentionConfig& config);
    
    /**
     * @brief Reset to default configuration
     */
    void reset_config();

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Configuration
    AttentionConfig config_;
    AttentionMode current_mode_;
    bool is_initialized_;
    
    // Module management
    std::vector<std::string> module_names_;
    std::map<std::string, ModuleAttentionState> module_states_;
    std::map<std::string, size_t> module_indices_;
    
    // Attention state
    std::vector<float> current_context_;
    std::vector<float> context_features_;
    std::map<std::string, float> context_priorities_;
    std::map<std::string, float> task_priorities_;
    
    // Temporal dynamics
    std::chrono::steady_clock::time_point last_update_time_;
    std::map<std::string, std::chrono::steady_clock::time_point> boost_end_times_;
    std::map<std::string, float> boost_amounts_;
    
    // Performance tracking
    std::map<std::string, std::vector<float>> performance_history_;
    size_t performance_history_size_;
    
    // Thread safety
    mutable std::mutex attention_mutex_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    // Attention computation
    void compute_context_based_weights();
    void compute_performance_based_weights();
    void apply_competitive_inhibition();
    void apply_temporal_dynamics(float dt);
    void apply_attention_boosts();
    
    // NLP-specific computations
    float compute_language_attention_weight(const std::string& module_name);
    float compute_reasoning_attention_weight(const std::string& module_name);
    void adjust_for_nlp_pipeline();
    
    // Utility methods
    float sigmoid(float x) const;
    float compute_entropy(const std::vector<float>& weights) const;
    float compute_variance(const std::vector<float>& values) const;
    void clamp_attention_weights();
    void log_attention_change(const std::string& module_name, float old_weight, float new_weight);
    
    // Validation
    bool validate_module_name(const std::string& module_name) const;
    bool validate_weight(float weight) const;
};

#endif // ATTENTION_CONTROLLER_H