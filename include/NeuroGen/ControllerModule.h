// ============================================================================
// NEUROMODULATORY CONTROLLER MODULE
// File: include/NeuroGen/ControllerModule.h
// ============================================================================

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <chrono>
#include <mutex>
#include <queue>

#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"

// ============================================================================
// NEUROTRANSMITTER AND NEUROMODULATOR DEFINITIONS
// ============================================================================

enum class NeuromodulatorType {
    DOPAMINE,           // Reward prediction, motivation, learning
    SEROTONIN,          // Mood regulation, plasticity modulation
    NOREPINEPHRINE,     // Attention, arousal, stress response
    ACETYLCHOLINE,      // Attention, learning, memory consolidation
    GABA,               // Inhibition, anxiety regulation
    GLUTAMATE,          // Excitation, synaptic plasticity
    OXYTOCIN,           // Social bonding, trust
    ENDORPHINS,         // Pain relief, reward enhancement
    CORTISOL,           // Stress response, memory modulation
    ADENOSINE           // Sleep pressure, metabolic regulation
};

enum class RewardSignalType {
    INTRINSIC_CURIOSITY,    // Internal exploration reward
    EXTRINSIC_TASK,         // External task completion reward
    SOCIAL_COOPERATION,     // Multi-module coordination reward
    EFFICIENCY_BONUS,       // Energy-efficient operation reward
    NOVELTY_DETECTION,      // New pattern discovery reward
    PREDICTION_ACCURACY,    // Successful prediction reward
    HOMEOSTATIC_BALANCE,    // Maintaining optimal states reward
    CREATIVITY_BURST        // Novel solution generation reward
};

struct NeuromodulatorState {
    NeuromodulatorType type;
    float concentration;        // Current concentration (0.0 - 1.0)
    float baseline_level;       // Homeostatic baseline
    float production_rate;      // Rate of synthesis
    float degradation_rate;     // Rate of breakdown
    float half_life;           // Time to half concentration
    float target_concentration; // Desired concentration level
    bool auto_regulate;        // Enable automatic homeostatic control
    
    NeuromodulatorState(NeuromodulatorType t) : type(t), concentration(0.3f), 
        baseline_level(0.3f), production_rate(0.1f), degradation_rate(0.05f),
        half_life(10.0f), target_concentration(0.3f), auto_regulate(true) {}
    
    void reset_to_defaults() {
        concentration = baseline_level;
        target_concentration = baseline_level;
    }
    
    void apply_stimulus(float intensity) {
        concentration = std::min(1.0f, concentration + intensity * production_rate);
    }
};

struct RewardSignal {
    RewardSignalType signal_type;  // Type of reward signal
    float magnitude;               // Strength of reward signal (0.0 - 1.0)
    float persistence;             // How long signal persists
    std::string source_module;     // Module that generated this reward
    std::string context;           // Context in which reward was generated
    std::chrono::steady_clock::time_point timestamp; // When this reward was generated
    
    RewardSignal() : signal_type(RewardSignalType::INTRINSIC_CURIOSITY), 
        magnitude(0.0f), persistence(1.0f), source_module(""), context(""),
        timestamp(std::chrono::steady_clock::now()) {}
    
    RewardSignal(RewardSignalType t, float mag, const std::string& src = "") 
        : signal_type(t), magnitude(mag), persistence(1.0f), source_module(src), 
          context(""), timestamp(std::chrono::steady_clock::now()) {}
};

struct NeuromodulationCommand {
    NeuromodulatorType modulator;
    float intensity;
    std::string target_module;  // Empty means global application
    float duration;            // How long to apply
    bool immediate;            // Apply immediately vs queue
    
    NeuromodulationCommand(NeuromodulatorType mod, float intens, 
                          const std::string& target = "", float dur = 1.0f)
        : modulator(mod), intensity(intens), target_module(target), 
          duration(dur), immediate(false) {}
};

struct ControllerConfig {
    // Initial neuromodulator levels
    float initial_dopamine_level = 0.3f;
    float initial_serotonin_level = 0.4f;
    float initial_norepinephrine_level = 0.25f;
    float initial_acetylcholine_level = 0.35f;
    float initial_gaba_level = 0.45f;
    float initial_glutamate_level = 0.4f;
    
    // Modulation parameters
    float reward_learning_rate = 0.01f;
    float stress_response_threshold = 0.7f;
    float attention_focus_strength = 0.6f;
    float homeostatic_correction_rate = 0.05f;
    
    // Performance tracking
    bool enable_detailed_logging = false;
    bool enable_auto_regulation = true;
    float performance_update_interval = 1.0f;
};

// ============================================================================
// CONTROLLER MODULE CLASS
// ============================================================================

class ControllerModule {
public:
    explicit ControllerModule(const ControllerConfig& config = ControllerConfig{});
    virtual ~ControllerModule();
    
    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    void register_module(const std::string& name, std::shared_ptr<NeuralModule> module);
    void unregister_module(const std::string& name);
    std::shared_ptr<NeuralModule> get_module(const std::string& name) const;
    std::vector<std::string> get_registered_modules() const;
    
    // ========================================================================
    // CORE CONTROL FUNCTIONS
    // ========================================================================
    
    void update(float dt);
    void reset();
    void emergency_stop();
    
    // ========================================================================
    // NEUROMODULATOR MANAGEMENT
    // ========================================================================
    
    void release_neuromodulator(NeuromodulatorType type, float intensity, 
                               const std::string& target_module = "");
    float get_concentration(NeuromodulatorType type) const;
    void set_baseline_level(NeuromodulatorType type, float level);
    void enable_auto_regulation(NeuromodulatorType type, bool enable);
    
    // ========================================================================
    // REWARD AND LEARNING SYSTEM
    // ========================================================================
    
    void apply_reward(const std::string& module_name, float reward_magnitude, 
                      RewardSignalType type = RewardSignalType::EXTRINSIC_TASK);
    void apply_punishment(const std::string& module_name, float punishment_magnitude);
    void generate_intrinsic_reward(const std::string& module_name, float novelty_factor);
    void update_performance_metrics();
    
    // ========================================================================
    // HIGH-LEVEL BEHAVIOR MODES
    // ========================================================================
    
    void enable_creative_mode(float intensity = 0.7f);
    void enable_focus_mode(const std::string& target_module, float intensity = 0.8f);
    void enable_exploration_mode(float curiosity_boost = 0.6f);
    void enable_consolidation_mode(float memory_strength = 0.7f);
    void enable_stress_response(float stress_level);
    void coordinate_module_activities();
    void apply_global_inhibition(float strength);
    
    // ========================================================================
    // MONITORING AND DIAGNOSTICS
    // ========================================================================
    
    std::string generate_status_report();
    void enable_detailed_logging(bool enable);
    float get_system_coherence() const { return system_coherence_level_; }
    float get_performance_trend() const { return global_performance_trend_; }
    float get_simulation_time() const { return simulation_time_; }
    
    // Advanced functionality
    void queue_command(const NeuromodulationCommand& command);
    std::vector<NeuromodulationCommand> generate_coordinated_response();
    float calculate_overall_system_performance();

private:
    // Configuration and state
    ControllerConfig config_;
    mutable std::mutex modules_mutex_;
    
    // Module registry
    std::unordered_map<std::string, std::shared_ptr<NeuralModule>> registered_modules_;
    std::unordered_map<std::string, float> module_performance_history_;
    std::unordered_map<std::string, float> attention_weights_;
    
    // Neuromodulator system
    std::unordered_map<NeuromodulatorType, std::unique_ptr<NeuromodulatorState>> neuromodulators_;
    std::queue<RewardSignal> pending_rewards_;
    std::queue<NeuromodulationCommand> pending_commands_;
    
    // Timing and performance
    float simulation_time_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    // Performance metrics
    std::unordered_map<std::string, NetworkStats> module_stats_history_;
    float global_performance_trend_;
    float system_coherence_level_;
    float neuron_activity_ratio_;  // Track overall neural activity level
    
    // State flags
    bool is_running_;
    bool detailed_logging_enabled_;
    float creativity_mode_factor_;
    
    // Action history for debugging
    std::vector<std::string> action_history_;
    
    // Reward signal storage
    std::vector<RewardSignal> reward_history_;
    
    // Performance metrics structure
    struct SystemPerformanceMetrics {
        float overall_performance = 0.5f;
        float learning_efficiency = 0.5f;
        float network_stability = 0.5f;
        float resource_utilization = 0.5f;
    } system_performance_metrics_;
    
    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================
    
    // Core internal methods
    void initialize_neuromodulators();
    void update_neuromodulator_dynamics(float dt);
    void process_pending_rewards();
    void execute_pending_commands();
    void assess_system_state();
    void generate_automatic_responses();
    void log_action(const std::string& action);
    
    // Helper functions
    float calculate_stress_level() const;
    float calculate_attention_demand() const;
    float calculate_learning_opportunity() const;
    float calculate_learning_efficiency();
    bool should_trigger_homeostatic_response() const;
    
    // **FIXED: Added missing declaration**
    void apply_neuromodulator_to_module(std::shared_ptr<NeuralModule> module, 
                                       NeuromodulatorType type, float intensity);
    
    // Specialized control algorithms
    void dopamine_reward_prediction_update(const RewardSignal& signal);
    void serotonin_mood_regulation();
    void norepinephrine_attention_modulation();
    void acetylcholine_learning_enhancement();
    void gaba_inhibitory_balance();
    void glutamate_excitatory_drive();
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Convert enum to string for logging and debugging
std::string to_string(NeuromodulatorType type);
std::string to_string(RewardSignalType type);

// Factory function for creating pre-configured controllers
std::unique_ptr<ControllerModule> create_learning_focused_controller();
std::unique_ptr<ControllerModule> create_exploration_focused_controller();
std::unique_ptr<ControllerModule> create_balanced_controller();
std::unique_ptr<ControllerModule> create_performance_focused_controller();

// Helper for module interconnection
void setup_standard_module_connections(ControllerModule& controller,
                                      std::shared_ptr<NeuralModule> perception,
                                      std::shared_ptr<NeuralModule> planning,
                                      std::shared_ptr<NeuralModule> motor);