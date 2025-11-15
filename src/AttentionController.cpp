// ============================================================================
// ATTENTION CONTROLLER IMPLEMENTATION - NLP-FOCUSED ARCHITECTURE
// File: src/AttentionController.cpp
// ============================================================================

#include "NeuroGen/AttentionController.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iomanip>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

AttentionController::AttentionController() 
    : current_mode_(AttentionMode::ADAPTIVE),
      is_initialized_(false),
      performance_history_size_(100),
      last_update_time_(std::chrono::steady_clock::now()) {
    
    // Initialize with default configuration
    config_.attention_decay_rate = 0.95f;
    config_.attention_boost_threshold = 0.7f;
    config_.global_inhibition_strength = 0.1f;
    config_.plasticity_rate = 0.01f;
    config_.context_sensitivity = 0.5f;
    config_.language_attention_bias = 1.2f;
    config_.enable_competitive_inhibition = true;
    config_.enable_attention_plasticity = true;
    
    std::cout << "🧠 Attention Controller: Initializing attention system..." << std::endl;
}

AttentionController::AttentionController(const AttentionConfig& config) 
    : config_(config),
      current_mode_(AttentionMode::ADAPTIVE),
      is_initialized_(false),
      performance_history_size_(100),
      last_update_time_(std::chrono::steady_clock::now()) {
    
    std::cout << "🧠 Attention Controller: Initializing with custom configuration..." << std::endl;
}

bool AttentionController::initialize() {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (is_initialized_) {
        std::cout << "⚠️ Attention Controller: Already initialized" << std::endl;
        return true;
    }
    
    try {
        // Initialize context and feature vectors
        current_context_.clear();
        context_features_.clear();
        
        // Initialize performance tracking
        performance_history_.clear();
        
        // Initialize timing
        last_update_time_ = std::chrono::steady_clock::now();
        
        is_initialized_ = true;
        
        std::cout << "✅ Attention Controller: Initialization complete" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Attention Controller: Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void AttentionController::shutdown() {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    module_names_.clear();
    module_states_.clear();
    module_indices_.clear();
    context_priorities_.clear();
    task_priorities_.clear();
    performance_history_.clear();
    
    is_initialized_ = false;
    
    std::cout << "🔌 Attention Controller: Shutdown complete" << std::endl;
}

// ============================================================================
// MODULE REGISTRATION AND MANAGEMENT
// ============================================================================

bool AttentionController::register_module(const std::string& module_name, float baseline_weight) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (!validate_module_name(module_name)) {
        std::cerr << "❌ Invalid module name: " << module_name << std::endl;
        return false;
    }
    
    if (is_module_registered(module_name)) {
        std::cout << "⚠️ Module '" << module_name << "' already registered" << std::endl;
        return true;
    }
    
    // Add to module lists
    module_names_.push_back(module_name);
    module_indices_[module_name] = module_names_.size() - 1;
    
    // Initialize module state
    ModuleAttentionState state;
    state.module_name = module_name;
    state.current_weight = baseline_weight;
    state.baseline_weight = baseline_weight;
    state.activation_history = 0.0f;
    state.performance_score = 0.5f;
    state.last_update = std::chrono::steady_clock::now();
    state.is_active = true;
    state.attention_history.reserve(performance_history_size_);
    
    module_states_[module_name] = state;
    
    // Initialize performance history
    performance_history_[module_name] = std::vector<float>();
    performance_history_[module_name].reserve(performance_history_size_);
    
    std::cout << "✅ Attention Controller: Registered module '" << module_name 
              << "' with baseline weight " << baseline_weight 
              << " (total modules: " << module_names_.size() << ")" << std::endl;
    
    return true;
}

bool AttentionController::unregister_module(const std::string& module_name) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (!is_module_registered(module_name)) {
        std::cout << "⚠️ Module '" << module_name << "' not registered" << std::endl;
        return false;
    }
    
    // Remove from all containers
    auto it = std::find(module_names_.begin(), module_names_.end(), module_name);
    if (it != module_names_.end()) {
        module_names_.erase(it);
    }
    
    module_states_.erase(module_name);
    module_indices_.erase(module_name);
    performance_history_.erase(module_name);
    boost_end_times_.erase(module_name);
    boost_amounts_.erase(module_name);
    
    // Rebuild module indices
    module_indices_.clear();
    for (size_t i = 0; i < module_names_.size(); ++i) {
        module_indices_[module_names_[i]] = i;
    }
    
    std::cout << "🗑️ Attention Controller: Unregistered module '" << module_name << "'" << std::endl;
    return true;
}

bool AttentionController::is_module_registered(const std::string& module_name) const {
    return module_states_.find(module_name) != module_states_.end();
}

std::vector<std::string> AttentionController::get_registered_modules() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    return module_names_;
}

size_t AttentionController::get_module_count() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    return module_names_.size();
}

// ============================================================================
// ATTENTION WEIGHT MANAGEMENT
// ============================================================================

float AttentionController::get_attention_weight(const std::string& module_name) const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    auto it = module_states_.find(module_name);
    if (it != module_states_.end()) {
        return it->second.current_weight;
    }
    
    // Return default weight if module not found
    return 1.0f;
}

bool AttentionController::set_attention_weight(const std::string& module_name, float weight) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (!validate_weight(weight)) {
        std::cerr << "❌ Invalid attention weight: " << weight << std::endl;
        return false;
    }
    
    auto it = module_states_.find(module_name);
    if (it != module_states_.end()) {
        float old_weight = it->second.current_weight;
        it->second.current_weight = weight;
        it->second.last_update = std::chrono::steady_clock::now();
        
        // Add to history
        if (it->second.attention_history.size() >= performance_history_size_) {
            it->second.attention_history.erase(it->second.attention_history.begin());
        }
        it->second.attention_history.push_back(weight);
        
        log_attention_change(module_name, old_weight, weight);
        return true;
    }
    
    std::cerr << "❌ Module '" << module_name << "' not registered" << std::endl;
    return false;
}

std::vector<float> AttentionController::get_all_attention_weights() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    std::vector<float> weights;
    weights.reserve(module_names_.size());
    
    for (const std::string& module_name : module_names_) {
        auto it = module_states_.find(module_name);
        if (it != module_states_.end()) {
            weights.push_back(it->second.current_weight);
        } else {
            weights.push_back(1.0f); // Default weight
        }
    }
    
    return weights;
}

std::map<std::string, float> AttentionController::get_attention_weight_map() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    std::map<std::string, float> weight_map;
    for (const auto& [module_name, state] : module_states_) {
        weight_map[module_name] = state.current_weight;
    }
    
    return weight_map;
}

void AttentionController::reset_attention_weights() {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    for (auto& [module_name, state] : module_states_) {
        state.current_weight = state.baseline_weight;
        state.last_update = std::chrono::steady_clock::now();
    }
    
    std::cout << "🔄 Attention Controller: Reset all attention weights to baseline" << std::endl;
}

void AttentionController::normalize_attention_weights() {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (module_states_.empty()) return;
    
    // Calculate current sum
    float total_weight = 0.0f;
    for (const auto& [module_name, state] : module_states_) {
        total_weight += state.current_weight;
    }
    
    if (total_weight <= 0.0f) {
        // If all weights are zero or negative, reset to equal weights
        float equal_weight = 1.0f;
        for (auto& [module_name, state] : module_states_) {
            state.current_weight = equal_weight;
        }
        return;
    }
    
    // Normalize to sum to module count
    float target_sum = static_cast<float>(module_states_.size());
    float scale_factor = target_sum / total_weight;
    
    for (auto& [module_name, state] : module_states_) {
        state.current_weight *= scale_factor;
        state.last_update = std::chrono::steady_clock::now();
    }
    
    std::cout << "📊 Attention Controller: Normalized attention weights (scale factor: " 
              << scale_factor << ")" << std::endl;
}

// ============================================================================
// CONTEXT-BASED ATTENTION CONTROL
// ============================================================================

void AttentionController::update_context(const std::vector<float>& context_vector) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    current_context_ = context_vector;
    
    // Update context features with exponential moving average
    if (context_features_.size() != context_vector.size()) {
        context_features_.resize(context_vector.size(), 0.0f);
    }
    
    for (size_t i = 0; i < std::min(context_features_.size(), context_vector.size()); ++i) {
        context_features_[i] = context_features_[i] * 0.9f + context_vector[i] * 0.1f;
    }
    
    // Recompute attention weights based on new context
    compute_attention_weights();
}

void AttentionController::update_language_context(float text_complexity, 
                                                 float reasoning_demand, 
                                                 float response_urgency) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    // Create specialized language context vector
    std::vector<float> language_context = {
        text_complexity,     // How complex is the input text
        reasoning_demand,    // How much reasoning is required
        response_urgency,    // How urgent is the response
        config_.language_attention_bias  // Language processing bias
    };
    
    // Set context priorities for language processing
    context_priorities_["text_complexity"] = text_complexity;
    context_priorities_["reasoning_demand"] = reasoning_demand;
    context_priorities_["response_urgency"] = response_urgency;
    
    // Update context and recompute weights
    update_context(language_context);
    
    std::cout << "🔤 Attention Controller: Updated language context (complexity: " 
              << text_complexity << ", reasoning: " << reasoning_demand 
              << ", urgency: " << response_urgency << ")" << std::endl;
}

void AttentionController::set_task_priorities(const std::string& task_type, 
                                             const std::map<std::string, float>& priority_map) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    task_priorities_.clear();
    for (const auto& [module_name, priority] : priority_map) {
        if (is_module_registered(module_name)) {
            task_priorities_[module_name] = std::max(0.0f, std::min(2.0f, priority));
        }
    }
    
    // Recompute attention weights
    compute_attention_weights();
    
    std::cout << "📋 Attention Controller: Set task priorities for '" << task_type 
              << "' (" << priority_map.size() << " modules)" << std::endl;
}

void AttentionController::set_context_priority(const std::string& context, float priority) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    context_priorities_[context] = std::max(0.0f, std::min(2.0f, priority));
    
    std::cout << "🎯 Attention Controller: Set context priority '" << context 
              << "' = " << priority << std::endl;
}

std::map<std::string, float> AttentionController::get_context_priorities() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    return context_priorities_;
}

// ============================================================================
// ATTENTION MODES AND CONTROL
// ============================================================================

void AttentionController::set_attention_mode(AttentionMode mode) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    AttentionMode old_mode = current_mode_;
    current_mode_ = mode;
    
    // Apply mode-specific attention patterns
    switch (mode) {
        case AttentionMode::BALANCED:
            reset_attention_weights();
            break;
            
        case AttentionMode::LANGUAGE_FOCUSED:
            focus_on_language_processing(1.5f);
            break;
            
        case AttentionMode::REASONING_FOCUSED:
            if (is_module_registered("reasoning_module")) {
                set_attention_weight("reasoning_module", 1.8f);
            }
            break;
            
        case AttentionMode::INPUT_FOCUSED:
            if (is_module_registered("input_module")) {
                set_attention_weight("input_module", 1.6f);
            }
            break;
            
        case AttentionMode::OUTPUT_FOCUSED:
            if (is_module_registered("output_module")) {
                set_attention_weight("output_module", 1.6f);
            }
            break;
            
        case AttentionMode::ADAPTIVE:
            // Will be handled by compute_attention_weights()
            break;
    }
    
    // Recompute weights if not already done by mode-specific actions
    if (mode == AttentionMode::ADAPTIVE) {
        compute_attention_weights();
    }
    
    std::cout << "🔄 Attention Controller: Changed mode from " << static_cast<int>(old_mode) 
              << " to " << static_cast<int>(mode) << std::endl;
}

AttentionController::AttentionMode AttentionController::get_attention_mode() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    return current_mode_;
}

void AttentionController::apply_global_inhibition(float strength) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    config_.global_inhibition_strength = std::max(0.0f, std::min(1.0f, strength));
    
    // Apply inhibition immediately
    apply_competitive_inhibition();
    
    std::cout << "🚫 Attention Controller: Applied global inhibition (strength: " 
              << strength << ")" << std::endl;
}

void AttentionController::enable_competitive_inhibition(bool enable) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    config_.enable_competitive_inhibition = enable;
    
    std::cout << "⚔️ Attention Controller: Competitive inhibition " 
              << (enable ? "ENABLED" : "DISABLED") << std::endl;
}

void AttentionController::boost_module_attention(const std::string& module_name, 
                                                float boost_amount, 
                                                int duration_ms) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (!is_module_registered(module_name)) {
        std::cerr << "❌ Cannot boost unregistered module: " << module_name << std::endl;
        return;
    }
    
    // Set boost parameters
    boost_amounts_[module_name] = boost_amount;
    boost_end_times_[module_name] = std::chrono::steady_clock::now() + 
                                   std::chrono::milliseconds(duration_ms);
    
    // Apply boost immediately
    auto it = module_states_.find(module_name);
    if (it != module_states_.end()) {
        float old_weight = it->second.current_weight;
        it->second.current_weight *= boost_amount;
        log_attention_change(module_name, old_weight, it->second.current_weight);
    }
    
    std::cout << "🚀 Attention Controller: Boosted '" << module_name 
              << "' by " << boost_amount << "x for " << duration_ms << "ms" << std::endl;
}

// ============================================================================
// DYNAMIC ATTENTION COMPUTATION
// ============================================================================

void AttentionController::compute_attention_weights() {
    if (!is_initialized_) return;
    
    // Apply different computation strategies based on mode
    switch (current_mode_) {
        case AttentionMode::BALANCED:
            // Keep equal weights - no computation needed
            break;
            
        case AttentionMode::ADAPTIVE:
            compute_context_based_weights();
            compute_performance_based_weights();
            break;
            
        default:
            // Mode-specific patterns already applied in set_attention_mode
            break;
    }
    
    // Always apply competitive inhibition and temporal dynamics
    if (config_.enable_competitive_inhibition) {
        apply_competitive_inhibition();
    }
    
    apply_attention_boosts();
    clamp_attention_weights();
}

void AttentionController::update_attention_dynamics(float dt) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    apply_temporal_dynamics(dt);
    compute_attention_weights();
    
    last_update_time_ = std::chrono::steady_clock::now();
}

void AttentionController::apply_attention_plasticity(const std::map<std::string, float>& module_performance) {
    if (!config_.enable_attention_plasticity) return;
    
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    for (const auto& [module_name, performance] : module_performance) {
        if (is_module_registered(module_name)) {
            update_module_performance(module_name, performance);
            
            // Adjust baseline weight based on performance
            auto it = module_states_.find(module_name);
            if (it != module_states_.end()) {
                float performance_adjustment = (performance - 0.5f) * config_.plasticity_rate;
                it->second.baseline_weight += performance_adjustment;
                it->second.baseline_weight = std::max(0.1f, std::min(2.0f, it->second.baseline_weight));
            }
        }
    }
}

void AttentionController::update_module_performance(const std::string& module_name, float performance_score) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (!is_module_registered(module_name)) return;
    
    // Update performance history
    auto& history = performance_history_[module_name];
    if (history.size() >= performance_history_size_) {
        history.erase(history.begin());
    }
    history.push_back(performance_score);
    
    // Update module state
    auto it = module_states_.find(module_name);
    if (it != module_states_.end()) {
        it->second.performance_score = performance_score;
        it->second.last_update = std::chrono::steady_clock::now();
    }
}

// ============================================================================
// NLP-SPECIFIC ATTENTION METHODS
// ============================================================================

void AttentionController::configure_for_nlp() {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    // Set NLP-optimized configuration
    config_.language_attention_bias = 1.3f;
    config_.context_sensitivity = 0.7f;
    config_.attention_decay_rate = 0.98f; // Slower decay for language processing
    config_.plasticity_rate = 0.005f;     // More conservative plasticity
    
    // Set attention mode to language-focused
    current_mode_ = AttentionMode::LANGUAGE_FOCUSED;
    
    std::cout << "🔤 Attention Controller: Configured for NLP processing" << std::endl;
}

void AttentionController::set_nlp_attention_weights(float input_weight, 
                                                   float language_weight, 
                                                   float reasoning_weight, 
                                                   float output_weight) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    // Set weights for NLP pipeline modules
    if (is_module_registered("input_module")) {
        set_attention_weight("input_module", input_weight);
    }
    if (is_module_registered("language_processing")) {
        set_attention_weight("language_processing", language_weight);
    }
    if (is_module_registered("reasoning_module")) {
        set_attention_weight("reasoning_module", reasoning_weight);
    }
    if (is_module_registered("output_module")) {
        set_attention_weight("output_module", output_weight);
    }
    
    // Central controller gets moderate attention
    if (is_module_registered("central_controller")) {
        set_attention_weight("central_controller", 1.0f);
    }
    
    std::cout << "⚖️ Attention Controller: Set NLP attention weights (I:" << input_weight 
              << " L:" << language_weight << " R:" << reasoning_weight 
              << " O:" << output_weight << ")" << std::endl;
}

void AttentionController::adapt_for_language_task(const std::string& task_type) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    if (task_type == "comprehension") {
        // Focus on input and language processing
        set_nlp_attention_weights(1.2f, 1.5f, 0.8f, 0.6f);
        
    } else if (task_type == "generation") {
        // Focus on reasoning and output
        set_nlp_attention_weights(0.8f, 1.0f, 1.3f, 1.4f);
        
    } else if (task_type == "reasoning") {
        // Focus on reasoning module
        set_nlp_attention_weights(0.7f, 1.1f, 1.6f, 0.9f);
        
    } else if (task_type == "conversation") {
        // Balanced attention for interactive dialogue
        set_nlp_attention_weights(1.0f, 1.2f, 1.1f, 1.0f);
        
    } else {
        // Default balanced weights
        set_nlp_attention_weights(1.0f, 1.0f, 1.0f, 1.0f);
    }
    
    std::cout << "🎯 Attention Controller: Adapted for language task '" << task_type << "'" << std::endl;
}

void AttentionController::focus_on_language_processing(float focus_strength) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    // Boost language processing modules
    if (is_module_registered("language_processing")) {
        set_attention_weight("language_processing", focus_strength);
    }
    if (is_module_registered("reasoning_module")) {
        set_attention_weight("reasoning_module", focus_strength * 0.8f);
    }
    
    // Reduce attention to other modules
    if (is_module_registered("input_module")) {
        set_attention_weight("input_module", 1.0f / focus_strength);
    }
    if (is_module_registered("output_module")) {
        set_attention_weight("output_module", 1.0f / focus_strength);
    }
    
    std::cout << "🎯 Attention Controller: Focused on language processing (strength: " 
              << focus_strength << ")" << std::endl;
}

// ============================================================================
// MONITORING AND ANALYSIS
// ============================================================================

AttentionController::ModuleAttentionState AttentionController::get_module_attention_state(const std::string& module_name) const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    auto it = module_states_.find(module_name);
    if (it != module_states_.end()) {
        return it->second;
    }
    
    // Return default state if module not found
    ModuleAttentionState default_state;
    default_state.module_name = module_name;
    default_state.is_active = false;
    return default_state;
}

std::map<std::string, float> AttentionController::get_attention_statistics() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    std::map<std::string, float> stats;
    
    if (module_states_.empty()) {
        return stats;
    }
    
    // Compute basic statistics
    std::vector<float> weights;
    for (const auto& [module_name, state] : module_states_) {
        weights.push_back(state.current_weight);
    }
    
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    float mean = sum / weights.size();
    
    float variance = 0.0f;
    for (float weight : weights) {
        variance += (weight - mean) * (weight - mean);
    }
    variance /= weights.size();
    
    float min_weight = *std::min_element(weights.begin(), weights.end());
    float max_weight = *std::max_element(weights.begin(), weights.end());
    
    stats["mean_attention"] = mean;
    stats["attention_variance"] = variance;
    stats["attention_std"] = std::sqrt(variance);
    stats["min_attention"] = min_weight;
    stats["max_attention"] = max_weight;
    stats["attention_range"] = max_weight - min_weight;
    stats["attention_entropy"] = compute_entropy(weights);
    stats["attention_focus"] = max_weight / sum; // Normalized max attention
    
    return stats;
}

float AttentionController::get_attention_entropy() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    std::vector<float> weights = get_all_attention_weights();
    return compute_entropy(weights);
}

float AttentionController::get_attention_focus() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    std::vector<float> weights = get_all_attention_weights();
    if (weights.empty()) return 0.0f;
    
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    float max_weight = *std::max_element(weights.begin(), weights.end());
    
    return (sum > 0.0f) ? (max_weight / sum) : 0.0f;
}

bool AttentionController::is_attention_stable() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    // Check if attention weights have stabilized (low variance in recent history)
    for (const auto& [module_name, state] : module_states_) {
        if (state.attention_history.size() < 10) {
            return false; // Not enough history
        }
        
        // Check variance of recent attention values
        auto recent_start = state.attention_history.end() - 10;
        std::vector<float> recent_weights(recent_start, state.attention_history.end());
        
        float variance = compute_variance(recent_weights);
        if (variance > 0.01f) { // Threshold for stability
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

AttentionController::AttentionConfig AttentionController::get_config() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    return config_;
}

void AttentionController::set_config(const AttentionConfig& config) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    config_ = config;
    
    std::cout << "⚙️ Attention Controller: Configuration updated" << std::endl;
}

void AttentionController::reset_config() {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    config_ = AttentionConfig(); // Use default configuration
    
    std::cout << "🔄 Attention Controller: Configuration reset to defaults" << std::endl;
}

// ============================================================================
// INTERNAL METHODS
// ============================================================================

void AttentionController::compute_context_based_weights() {
    if (current_context_.empty()) return;
    
    for (auto& [module_name, state] : module_states_) {
        float context_weight = compute_language_attention_weight(module_name);
        
        // Blend with current weight
        float blend_factor = config_.context_sensitivity;
        state.current_weight = (1.0f - blend_factor) * state.current_weight + 
                              blend_factor * context_weight;
    }
}

void AttentionController::compute_performance_based_weights() {
    for (auto& [module_name, state] : module_states_) {
        if (performance_history_[module_name].size() < 5) {
            continue; // Not enough performance data
        }
        
        // Compute recent performance average
        auto& history = performance_history_[module_name];
        auto recent_start = history.end() - std::min(static_cast<size_t>(10), history.size());
        float recent_performance = std::accumulate(recent_start, history.end(), 0.0f) / 
                                  std::distance(recent_start, history.end());
        
        // Adjust weight based on performance
        float performance_adjustment = (recent_performance - 0.5f) * config_.plasticity_rate;
        state.current_weight += performance_adjustment;
    }
}

void AttentionController::apply_competitive_inhibition() {
    if (module_states_.empty()) return;
    
    // Calculate total activation
    float total_attention = 0.0f;
    for (const auto& [module_name, state] : module_states_) {
        total_attention += state.current_weight;
    }
    
    // Apply inhibition factor
    float target_total = static_cast<float>(module_states_.size());
    float inhibition_factor = 1.0f - config_.global_inhibition_strength * 
                             (total_attention - target_total) / target_total;
    inhibition_factor = std::max(0.1f, std::min(inhibition_factor, 1.0f));
    
    // Apply inhibition to all modules
    for (auto& [module_name, state] : module_states_) {
        state.current_weight *= inhibition_factor;
    }
}

void AttentionController::apply_temporal_dynamics(float dt) {
    for (auto& [module_name, state] : module_states_) {
        // Apply attention decay towards baseline
        float decay_factor = std::pow(config_.attention_decay_rate, dt);
        state.current_weight = decay_factor * state.current_weight + 
                              (1.0f - decay_factor) * state.baseline_weight;
        
        // Update activation history
        state.activation_history = 0.9f * state.activation_history + 0.1f * state.current_weight;
    }
}

void AttentionController::apply_attention_boosts() {
    auto current_time = std::chrono::steady_clock::now();
    
    // Check and apply active boosts
    for (auto it = boost_end_times_.begin(); it != boost_end_times_.end();) {
        const std::string& module_name = it->first;
        
        if (current_time >= it->second) {
            // Boost expired - remove boost
            auto boost_it = boost_amounts_.find(module_name);
            if (boost_it != boost_amounts_.end()) {
                auto state_it = module_states_.find(module_name);
                if (state_it != module_states_.end()) {
                    state_it->second.current_weight /= boost_it->second; // Remove boost
                }
                boost_amounts_.erase(boost_it);
            }
            it = boost_end_times_.erase(it);
        } else {
            ++it;
        }
    }
}

float AttentionController::compute_language_attention_weight(const std::string& module_name) {
    float base_weight = 1.0f;
    
    // Module-specific attention computation for NLP
    if (module_name == "central_controller") {
        // Central controller gets stable moderate attention
        base_weight = 1.0f;
        
    } else if (module_name == "input_module") {
        // Input module attention based on input complexity
        float complexity = context_priorities_.count("text_complexity") ? 
                          context_priorities_.at("text_complexity") : 0.5f;
        base_weight = 0.7f + 0.5f * complexity;
        
    } else if (module_name == "language_processing") {
        // Language processing gets highest base attention
        base_weight = config_.language_attention_bias;
        
    } else if (module_name == "reasoning_module") {
        // Reasoning attention based on reasoning demand
        float demand = context_priorities_.count("reasoning_demand") ? 
                      context_priorities_.at("reasoning_demand") : 0.5f;
        base_weight = 0.8f + 0.6f * demand;
        
    } else if (module_name == "output_module") {
        // Output attention based on response urgency
        float urgency = context_priorities_.count("response_urgency") ? 
                       context_priorities_.at("response_urgency") : 0.5f;
        base_weight = 0.6f + 0.5f * urgency;
    }
    
    return base_weight;
}

float AttentionController::compute_reasoning_attention_weight(const std::string& module_name) {
    // Boost reasoning-related modules
    if (module_name == "reasoning_module") {
        return 1.5f;
    } else if (module_name == "language_processing") {
        return 1.2f; // Language processing supports reasoning
    } else {
        return 0.8f; // Reduce other modules
    }
}

void AttentionController::adjust_for_nlp_pipeline() {
    // Ensure proper attention flow through NLP pipeline
    // Input -> Language -> Reasoning -> Output
    
    const std::vector<std::string> pipeline_order = {
        "input_module", "language_processing", "reasoning_module", "output_module"
    };
    
    for (size_t i = 0; i < pipeline_order.size(); ++i) {
        const std::string& module_name = pipeline_order[i];
        if (is_module_registered(module_name)) {
            // Attention flows through pipeline
            float pipeline_weight = 0.8f + 0.4f * std::sin(i * M_PI / (pipeline_order.size() - 1));
            
            auto it = module_states_.find(module_name);
            if (it != module_states_.end()) {
                it->second.current_weight = 0.7f * it->second.current_weight + 
                                           0.3f * pipeline_weight;
            }
        }
    }
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

float AttentionController::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

float AttentionController::compute_entropy(const std::vector<float>& weights) const {
    if (weights.empty()) return 0.0f;
    
    // Normalize weights to probabilities
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum <= 0.0f) return 0.0f;
    
    float entropy = 0.0f;
    for (float weight : weights) {
        float prob = weight / sum;
        if (prob > 0.0f) {
            entropy -= prob * std::log2(prob);
        }
    }
    
    return entropy;
}

float AttentionController::compute_variance(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;
    
    float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    float variance = 0.0f;
    
    for (float value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    return variance / values.size();
}

void AttentionController::clamp_attention_weights() {
    for (auto& [module_name, state] : module_states_) {
        state.current_weight = std::max(0.1f, std::min(2.0f, state.current_weight));
    }
}

void AttentionController::log_attention_change(const std::string& module_name, 
                                              float old_weight, 
                                              float new_weight) {
    if (std::abs(new_weight - old_weight) > 0.1f) {
        std::cout << "🎯 Attention: " << module_name << " " 
                  << std::fixed << std::setprecision(2) 
                  << old_weight << " → " << new_weight << std::endl;
    }
}

bool AttentionController::validate_module_name(const std::string& module_name) const {
    return !module_name.empty() && module_name.length() < 100;
}

bool AttentionController::validate_weight(float weight) const {
    return weight >= 0.0f && weight <= 5.0f && !std::isnan(weight) && !std::isinf(weight);
}