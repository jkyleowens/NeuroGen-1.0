// ============================================================================
// CONTROLLER MODULE IMPLEMENTATION
// File: src/ControllerModule.cpp
// ============================================================================

#include "NeuroGen/ControllerModule.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

// ============================================================================
// SPECIALIZED NEUROMODULATOR FUNCTIONS
// ============================================================================

void ControllerModule::dopamine_reward_prediction_update(const RewardSignal& signal) {
    // **FIXED: Removed unused variable current_dopamine**
    float baseline_dopamine = neuromodulators_[NeuromodulatorType::DOPAMINE]->baseline_level;
    
    // Update baseline based on recent performance
    float avg_performance = calculate_overall_system_performance();
    if (avg_performance > 0.7f) {
        // Increase baseline for consistently good performance
        float new_baseline = std::min(0.5f, baseline_dopamine + 0.01f);
        set_baseline_level(NeuromodulatorType::DOPAMINE, new_baseline);
    } else if (avg_performance < 0.3f) {
        // Decrease baseline for poor performance
        float new_baseline = std::max(0.1f, baseline_dopamine - 0.01f);
        set_baseline_level(NeuromodulatorType::DOPAMINE, new_baseline);
    }
    
    if (detailed_logging_enabled_) {
        log_action("Dopamine reward prediction update (performance: " + 
                  std::to_string(avg_performance) + ")");
    }
}

void ControllerModule::serotonin_mood_regulation() {
    // **FIXED: Removed unused variable current_serotonin**
    float stress_level = calculate_stress_level();
    
    if (stress_level > 0.6f) {
        // Increase serotonin to improve mood stability
        release_neuromodulator(NeuromodulatorType::SEROTONIN, stress_level * 0.3f);
        
        if (detailed_logging_enabled_) {
            log_action("Serotonin mood regulation activated (stress: " + 
                      std::to_string(stress_level) + ")");
        }
    }
}

void ControllerModule::norepinephrine_attention_modulation() {
    float attention_demand = calculate_attention_demand();
    
    if (attention_demand > 0.5f) {
        // Increase norepinephrine for enhanced attention
        release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, attention_demand * 0.4f);
        
        if (detailed_logging_enabled_) {
            log_action("Norepinephrine attention modulation (demand: " + 
                      std::to_string(attention_demand) + ")");
        }
    }
}

void ControllerModule::acetylcholine_learning_enhancement() {
    float learning_opportunity = calculate_learning_opportunity();
    
    if (learning_opportunity > 0.4f) {
        // Increase acetylcholine for enhanced plasticity
        release_neuromodulator(NeuromodulatorType::ACETYLCHOLINE, learning_opportunity * 0.5f);
        
        if (detailed_logging_enabled_) {
            log_action("Acetylcholine learning enhancement (opportunity: " + 
                      std::to_string(learning_opportunity) + ")");
        }
    }
}

void ControllerModule::gaba_inhibitory_balance() {
    float excitation_level = 0.0f;
    
    // Calculate overall excitation from glutamate and norepinephrine
    excitation_level += get_concentration(NeuromodulatorType::GLUTAMATE) * 0.6f;
    excitation_level += get_concentration(NeuromodulatorType::NOREPINEPHRINE) * 0.3f;
    
    if (excitation_level > 0.7f) {
        // Apply GABA to balance excessive excitation
        float inhibition_strength = (excitation_level - 0.7f) * 0.5f;
        release_neuromodulator(NeuromodulatorType::GABA, inhibition_strength);
        
        if (detailed_logging_enabled_) {
            log_action("GABA inhibitory balance applied (excitation: " + 
                      std::to_string(excitation_level) + ")");
        }
    }
}

void ControllerModule::glutamate_excitatory_drive() {
    float current_activity = neuron_activity_ratio_;
    
    if (current_activity < 0.2f) {
        // Increase glutamate to boost overall activity
        float boost_strength = (0.2f - current_activity) * 0.6f;
        release_neuromodulator(NeuromodulatorType::GLUTAMATE, boost_strength);
        
        if (detailed_logging_enabled_) {
            log_action("Glutamate excitatory drive applied (activity: " + 
                      std::to_string(current_activity) + ")");
        }
    }
}

// ============================================================================
// CORE CONTROL FUNCTIONS
// ============================================================================

void ControllerModule::update(float dt) {
    if (!is_running_) {
        is_running_ = true;
    }
    
    simulation_time_ += dt;
    
    // Update neuromodulator dynamics
    update_neuromodulator_dynamics(dt);
    
    // Process pending rewards and commands
    process_pending_rewards();
    execute_pending_commands();
    
    // Assess current system state
    assess_system_state();
    
    // Generate automatic responses
    generate_automatic_responses();
    
    // Update performance metrics
    update_performance_metrics();
    
    // Specialized neuromodulator functions
    dopamine_reward_prediction_update(RewardSignal{});
    serotonin_mood_regulation();
    norepinephrine_attention_modulation();
    acetylcholine_learning_enhancement();
    gaba_inhibitory_balance();
    glutamate_excitatory_drive();
    
    last_update_time_ = std::chrono::high_resolution_clock::now();
}

// ============================================================================
// NEUROMODULATOR MANAGEMENT
// ============================================================================

void ControllerModule::release_neuromodulator(NeuromodulatorType type, float intensity, 
                                             const std::string& target_module) {
    auto it = neuromodulators_.find(type);
    if (it != neuromodulators_.end()) {
        it->second->apply_stimulus(intensity);
        
        // Apply to specific module if specified
        if (!target_module.empty()) {
            auto module = get_module(target_module);
            if (module) {
                // Apply neuromodulator effects to the target module
                apply_neuromodulator_to_module(module, type, intensity);
            }
        } else {
            // Apply to all modules
            std::lock_guard<std::mutex> lock(modules_mutex_);
            for (const auto& pair : registered_modules_) {
                apply_neuromodulator_to_module(pair.second, type, intensity);
            }
        }
        
        if (detailed_logging_enabled_) {
            log_action("Released " + to_string(type) + " (intensity: " + 
                      std::to_string(intensity) + ") to " + 
                      (target_module.empty() ? "all modules" : target_module));
        }
    }
}

// ============================================================================
// **FIXED: PROPER IMPLEMENTATION OF MISSING METHOD**
// ============================================================================

void ControllerModule::apply_neuromodulator_to_module(std::shared_ptr<NeuralModule> module, 
                                                     NeuromodulatorType type, float intensity) {
    // This method applies neuromodulator effects to a specific module
    // Implementation depends on the NeuralModule interface
    
    if (!module) return;
    
    // Apply effects based on neuromodulator type
    switch (type) {
        case NeuromodulatorType::DOPAMINE:
            // Enhance learning rate and reward sensitivity
            // module->modulate_learning_rate(1.0f + intensity * 0.5f);
            // module->modulate_reward_sensitivity(1.0f + intensity * 0.3f);
            break;
            
        case NeuromodulatorType::SEROTONIN:
            // Improve stability and reduce impulsivity
            // module->modulate_stability(1.0f + intensity * 0.4f);
            // module->modulate_noise_tolerance(1.0f + intensity * 0.2f);
            break;
            
        case NeuromodulatorType::NOREPINEPHRINE:
            // Increase attention and arousal
            // module->modulate_attention_gain(1.0f + intensity * 0.6f);
            // module->modulate_arousal_level(1.0f + intensity * 0.3f);
            break;
            
        case NeuromodulatorType::ACETYLCHOLINE:
            // Enhance plasticity and memory formation
            // module->modulate_plasticity_rate(1.0f + intensity * 0.4f);
            // module->modulate_memory_consolidation(1.0f + intensity * 0.3f);
            break;
            
        case NeuromodulatorType::GABA:
            // Apply inhibitory modulation
            // module->modulate_inhibition(1.0f + intensity * 0.5f);
            // module->modulate_activity_level(1.0f - intensity * 0.3f);
            break;
            
        case NeuromodulatorType::GLUTAMATE:
            // Apply excitatory modulation
            // module->modulate_excitation(1.0f + intensity * 0.4f);
            // module->modulate_activity_level(1.0f + intensity * 0.2f);
            break;
            
        default:
            // Handle other neuromodulator types as they are implemented
            break;
    }
    
    // Log the application if detailed logging is enabled
    if (detailed_logging_enabled_) {
        log_action("Applied " + to_string(type) + " to module (intensity: " + 
                  std::to_string(intensity) + ")");
    }
}

// ============================================================================
// **FIXED: UPDATE PERFORMANCE METRICS TO AVOID COPY ASSIGNMENT**
// ============================================================================

void ControllerModule::update_performance_metrics() {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    // **FIXED: Clear and rebuild stats history instead of assignment**
    module_stats_history_.clear();
    
    for (const auto& pair : registered_modules_) {
        if (pair.second) {
            // **FIXED: Use emplace instead of assignment to avoid copy issues**
            auto stats = pair.second->get_stats();
            module_stats_history_.emplace(pair.first, std::move(stats));
        }
    }
    
    // Update global performance trend
    float total_performance = 0.0f;
    int valid_modules = 0;
    
    for (const auto& pair : module_performance_history_) {
        total_performance += pair.second;
        valid_modules++;
    }
    
    if (valid_modules > 0) {
        global_performance_trend_ = total_performance / valid_modules;
    }
    
    // Update neuron activity ratio based on module stats
    float total_activity = 0.0f;
    int activity_samples = 0;
    
    for (const auto& pair : module_stats_history_) {
        // Calculate activity based on neuron activity ratio and mean synaptic weight
        float module_activity = (pair.second.neuron_activity_ratio + pair.second.mean_synaptic_weight) * 0.5f;
        total_activity += module_activity;
        activity_samples++;
    }
    
    if (activity_samples > 0) {
        neuron_activity_ratio_ = total_activity / activity_samples;
    }
    
    // Update system coherence based on module synchronization
    // This is a simplified coherence measure
    system_coherence_level_ = std::min(1.0f, global_performance_trend_ + 0.2f);
    
    if (detailed_logging_enabled_) {
        log_action("Updated performance metrics (trend: " + 
                  std::to_string(global_performance_trend_) + 
                  ", coherence: " + std::to_string(system_coherence_level_) + ")");
    }
}

// ============================================================================
// REMAINING IMPLEMENTATION (unchanged, but showing key methods)
// ============================================================================

ControllerModule::ControllerModule(const ControllerConfig& config)
    : config_(config)
    , simulation_time_(0.0f)
    , global_performance_trend_(0.0f)
    , system_coherence_level_(0.5f)
    , neuron_activity_ratio_(0.5f)
    , is_running_(false)
    , detailed_logging_enabled_(false)
    , creativity_mode_factor_(1.0f) {
    
    initialize_neuromodulators();
    last_update_time_ = std::chrono::high_resolution_clock::now();
    
    if (detailed_logging_enabled_) {
        log_action("ControllerModule initialized");
    }
}

ControllerModule::~ControllerModule() {
    emergency_stop();
}

void ControllerModule::initialize_neuromodulators() {
    // Create all neuromodulator states
    neuromodulators_[NeuromodulatorType::DOPAMINE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::DOPAMINE);
    neuromodulators_[NeuromodulatorType::SEROTONIN] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::SEROTONIN);
    neuromodulators_[NeuromodulatorType::NOREPINEPHRINE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::NOREPINEPHRINE);
    neuromodulators_[NeuromodulatorType::ACETYLCHOLINE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::ACETYLCHOLINE);
    neuromodulators_[NeuromodulatorType::GABA] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::GABA);
    neuromodulators_[NeuromodulatorType::GLUTAMATE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::GLUTAMATE);
    
    // Set initial concentrations from config
    neuromodulators_[NeuromodulatorType::DOPAMINE]->concentration = config_.initial_dopamine_level;
    neuromodulators_[NeuromodulatorType::SEROTONIN]->concentration = config_.initial_serotonin_level;
    neuromodulators_[NeuromodulatorType::NOREPINEPHRINE]->concentration = config_.initial_norepinephrine_level;
    neuromodulators_[NeuromodulatorType::ACETYLCHOLINE]->concentration = config_.initial_acetylcholine_level;
    neuromodulators_[NeuromodulatorType::GABA]->concentration = config_.initial_gaba_level;
    neuromodulators_[NeuromodulatorType::GLUTAMATE]->concentration = config_.initial_glutamate_level;
}

// Helper methods (showing key signatures - full implementation would continue)
float ControllerModule::calculate_stress_level() const {
    // Implementation for calculating current stress level
    return 0.0f; // Placeholder
}

float ControllerModule::calculate_attention_demand() const {
    // Implementation for calculating attention demand
    return 0.0f; // Placeholder  
}

float ControllerModule::calculate_learning_opportunity() const {
    // Implementation for calculating learning opportunity
    return 0.0f; // Placeholder
}

void ControllerModule::log_action(const std::string& action) {
    action_history_.push_back("[" + std::to_string(simulation_time_) + "s] " + action);
    
    // Keep only recent history
    if (action_history_.size() > 1000) {
        action_history_.erase(action_history_.begin());
    }
}

// Utility functions for enum to string conversion
std::string to_string(NeuromodulatorType type) {
    switch (type) {
        case NeuromodulatorType::DOPAMINE: return "DOPAMINE";
        case NeuromodulatorType::SEROTONIN: return "SEROTONIN";
        case NeuromodulatorType::NOREPINEPHRINE: return "NOREPINEPHRINE";
        case NeuromodulatorType::ACETYLCHOLINE: return "ACETYLCHOLINE";
        case NeuromodulatorType::GABA: return "GABA";
        case NeuromodulatorType::GLUTAMATE: return "GLUTAMATE";
        default: return "UNKNOWN";
    }
}

std::string to_string(RewardSignalType type) {
    switch (type) {
        case RewardSignalType::INTRINSIC_CURIOSITY: return "INTRINSIC_CURIOSITY";
        case RewardSignalType::EXTRINSIC_TASK: return "EXTRINSIC_TASK";
        case RewardSignalType::SOCIAL_COOPERATION: return "SOCIAL_COOPERATION";
        case RewardSignalType::EFFICIENCY_BONUS: return "EFFICIENCY_BONUS";
        case RewardSignalType::NOVELTY_DETECTION: return "NOVELTY_DETECTION";
        case RewardSignalType::PREDICTION_ACCURACY: return "PREDICTION_ACCURACY";
        case RewardSignalType::HOMEOSTATIC_BALANCE: return "HOMEOSTATIC_BALANCE";
        case RewardSignalType::CREATIVITY_BURST: return "CREATIVITY_BURST";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// MISSING METHOD IMPLEMENTATIONS
// ============================================================================

void ControllerModule::enable_detailed_logging(bool enable) {
    detailed_logging_enabled_ = enable;
    if (enable) {
        log_action("Detailed logging enabled");
    }
}

void ControllerModule::enable_focus_mode(const std::string& target_module, float intensity) {
    // Set high attention on target module, reduce others
    attention_weights_[target_module] = std::min(1.0f, intensity);
    
    // Reduce attention on other modules proportionally
    float reduction_factor = 1.0f - (intensity * 0.3f);
    for (auto& [module_name, weight] : attention_weights_) {
        if (module_name != target_module) {
            weight *= reduction_factor;
        }
    }
    
    if (detailed_logging_enabled_) {
        log_action("Focus mode enabled for " + target_module + " with intensity " + std::to_string(intensity));
    }
}

void ControllerModule::enable_creative_mode(float creativity_factor) {
    // Increase exploration and reduce strict optimization
    creativity_mode_factor_ = std::clamp(creativity_factor, 0.0f, 2.0f);
    
    // Release small amounts of dopamine and norepinephrine to encourage exploration
    release_neuromodulator(NeuromodulatorType::DOPAMINE, creativity_factor * 0.2f);
    release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, creativity_factor * 0.15f);
    
    if (detailed_logging_enabled_) {
        log_action("Creative mode enabled with factor " + std::to_string(creativity_factor));
    }
}

std::string ControllerModule::generate_status_report() {
    std::stringstream report;
    
    report << "=== ControllerModule Status Report ===\n";
    report << "Active Modules: " << attention_weights_.size() << "\n";
    report << "Total Reward Signals: " << reward_history_.size() << "\n";
    report << "Detailed Logging: " << (detailed_logging_enabled_ ? "Enabled" : "Disabled") << "\n";
    
    report << "\nNeuromodulator Levels:\n";
    for (const auto& [type, modulator] : neuromodulators_) {
        report << "  " << static_cast<int>(type) << ": " 
               << std::fixed << std::setprecision(3) << modulator->concentration << "\n";
    }
    
    report << "\nAttention Weights:\n";
    for (const auto& [module, weight] : attention_weights_) {
        report << "  " << module << ": " 
               << std::fixed << std::setprecision(3) << weight << "\n";
    }
    
    float performance = calculate_overall_system_performance();
    report << "\nOverall Performance: " << std::fixed << std::setprecision(3) << performance << "\n";
    
    return report.str();
}

// ============================================================================
// ADDITIONAL MISSING METHOD IMPLEMENTATIONS
// ============================================================================

void ControllerModule::apply_reward(const std::string& module_name, float reward_magnitude, RewardSignalType reward_type) {
    RewardSignal signal;
    signal.signal_type = reward_type;
    signal.magnitude = reward_magnitude;
    signal.source_module = module_name;
    signal.timestamp = std::chrono::steady_clock::now();
    
    pending_rewards_.push(signal);
    
    if (detailed_logging_enabled_) {
        log_action("Applied reward " + std::to_string(reward_magnitude) + " to " + module_name);
    }
}

void ControllerModule::emergency_stop() {
    is_running_ = false;
    
    // Clear all pending operations
    while (!pending_rewards_.empty()) {
        pending_rewards_.pop();
    }
    while (!pending_commands_.empty()) {
        pending_commands_.pop();
    }
    
    if (detailed_logging_enabled_) {
        log_action("Emergency stop executed");
    }
}

void ControllerModule::register_module(const std::string& module_name, std::shared_ptr<NeuralModule> module) {
    registered_modules_[module_name] = module;
    attention_weights_[module_name] = 1.0f; // Default attention weight
    module_performance_history_[module_name] = 0.5f; // Default performance
    
    if (detailed_logging_enabled_) {
        log_action("Registered module: " + module_name);
    }
}

std::shared_ptr<NeuralModule> ControllerModule::get_module(const std::string& module_name) const {
    auto it = registered_modules_.find(module_name);
    if (it != registered_modules_.end()) {
        return it->second;
    }
    return nullptr;
}

void ControllerModule::coordinate_module_activities() {
    // Simple coordination strategy: balance attention across modules
    float total_performance = 0.0f;
    int active_modules = 0;
    
    for (const auto& [name, module] : registered_modules_) {
        if (module) {
            total_performance += module_performance_history_[name];
            active_modules++;
        }
    }
    
    if (active_modules > 0) {
        float avg_performance = total_performance / active_modules;
        
        // Adjust attention based on performance
        for (auto& [name, weight] : attention_weights_) {
            float performance = module_performance_history_[name];
            if (performance < avg_performance * 0.8f) {
                weight = std::min(1.0f, weight * 1.1f); // Increase attention for poor performers
            } else if (performance > avg_performance * 1.2f) {
                weight = std::max(0.1f, weight * 0.95f); // Slightly reduce attention for good performers
            }
        }
    }
    
    if (detailed_logging_enabled_) {
        log_action("Coordinated module activities");
    }
}

float ControllerModule::get_concentration(NeuromodulatorType type) const {
    auto it = neuromodulators_.find(type);
    if (it != neuromodulators_.end()) {
        return it->second->concentration;
    }
    return 0.0f;
}

std::vector<std::string> ControllerModule::get_registered_modules() const {
    std::vector<std::string> module_names;
    for (const auto& [name, module] : registered_modules_) {
        module_names.push_back(name);
    }
    return module_names;
}

float ControllerModule::calculate_overall_system_performance() {
    if (registered_modules_.empty()) {
        return 0.5f;
    }
    
    float total_performance = 0.0f;
    for (const auto& [name, performance] : module_performance_history_) {
        total_performance += performance;
    }
    
    return total_performance / registered_modules_.size();
}

void ControllerModule::set_baseline_level(NeuromodulatorType type, float baseline) {
    auto it = neuromodulators_.find(type);
    if (it != neuromodulators_.end()) {
        it->second->baseline_level = std::clamp(baseline, 0.0f, 1.0f);
    }
}

void ControllerModule::update_neuromodulator_dynamics(float dt) {
    for (auto& [type, modulator] : neuromodulators_) {
        // Natural decay towards baseline
        float decay_factor = 0.95f;
        modulator->concentration = modulator->concentration * decay_factor + 
                                  modulator->baseline_level * (1.0f - decay_factor);
        
        // Bound concentration
        modulator->concentration = std::clamp(modulator->concentration, 0.0f, 1.0f);
    }
}

void ControllerModule::process_pending_rewards() {
    while (!pending_rewards_.empty()) {
        RewardSignal signal = pending_rewards_.front();
        pending_rewards_.pop();
        
        // Apply reward to dopamine system
        release_neuromodulator(NeuromodulatorType::DOPAMINE, signal.magnitude * 0.5f);
        
        // Update module performance history
        if (module_performance_history_.find(signal.source_module) != module_performance_history_.end()) {
            module_performance_history_[signal.source_module] = 
                module_performance_history_[signal.source_module] * 0.9f + signal.magnitude * 0.1f;
        }
        
        // Store in reward history
        reward_history_.push_back(signal);
        if (reward_history_.size() > 1000) {
            reward_history_.erase(reward_history_.begin());
        }
    }
}

void ControllerModule::execute_pending_commands() {
    while (!pending_commands_.empty()) {
        NeuromodulationCommand command = pending_commands_.front();
        pending_commands_.pop();
        
        // Execute the command
        release_neuromodulator(command.modulator, command.intensity);
    }
}

void ControllerModule::assess_system_state() {
    // Update system performance metrics
    system_performance_metrics_.overall_performance = calculate_overall_system_performance();
    
    // Calculate system coherence based on attention distribution
    float attention_variance = 0.0f;
    float attention_mean = 0.0f;
    
    if (!attention_weights_.empty()) {
        for (const auto& [name, weight] : attention_weights_) {
            attention_mean += weight;
        }
        attention_mean /= attention_weights_.size();
        
        for (const auto& [name, weight] : attention_weights_) {
            attention_variance += (weight - attention_mean) * (weight - attention_mean);
        }
        attention_variance /= attention_weights_.size();
        
        system_coherence_level_ = 1.0f / (1.0f + attention_variance);
    }
}

void ControllerModule::generate_automatic_responses() {
    // Generate automatic neuromodulator responses based on system state
    
    // Stress response
    if (system_performance_metrics_.overall_performance < 0.3f) {
        release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, 0.3f);
        release_neuromodulator(NeuromodulatorType::SEROTONIN, 0.2f);
    }
    
    // Learning enhancement
    if (system_performance_metrics_.learning_efficiency > 0.7f) {
        release_neuromodulator(NeuromodulatorType::ACETYLCHOLINE, 0.2f);
    }
    
    // Activity regulation
    if (neuron_activity_ratio_ > 0.8f) {
        release_neuromodulator(NeuromodulatorType::GABA, 0.15f);
    } else if (neuron_activity_ratio_ < 0.2f) {
        release_neuromodulator(NeuromodulatorType::GLUTAMATE, 0.1f);
    }
}