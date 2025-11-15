// Update 2: Enhanced Neural Module Implementation
// File: src/EnhancedNeuralModule.cpp (UPDATE EXISTING)

#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/LearningState.h"
#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================================================
// ENHANCED LEARNING STATE IMPLEMENTATION
// ============================================================================

bool EnhancedNeuralModule::initialize() {
    // Call parent initialization
    if (!NeuralModule::initialize()) {
        return false;
    }
    
    // Initialize learning traces
    initializeLearningTraces();
    
    // Initialize performance tracking
    prediction_error_history_.resize(100, 0.0f);
    performance_history_index_ = 0;
    
    // Set initial consolidation time
    last_consolidation_ = std::chrono::steady_clock::now();
    
    std::cout << "✅ Enhanced neural module initialized: " << module_name_ << std::endl;
    return true;
}

void EnhancedNeuralModule::initializeLearningTraces() {
    // Initialize eligibility traces to zero
    size_t num_synapses = synaptic_weights_.size();
    eligibility_traces_.resize(num_synapses, 0.0f);
    synaptic_tags_.resize(num_synapses, 0.0f);
    
    // Initialize firing rate buffer (last 1000ms at 1Hz updates)
    firing_rate_buffer_.resize(1000, 0.0f);
    
    std::cout << "🧠 Initialized learning traces for " << num_synapses << " synapses" << std::endl;
}

ModuleLearningState EnhancedNeuralModule::getLearningState() const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    ModuleLearningState state;
    
    // Basic identification
    state.module_name = module_name_;
    state.module_id = std::hash<std::string>{}(module_name_);
    
    // Learning traces
    state.eligibility_traces = eligibility_traces_;
    state.synaptic_tags = synaptic_tags_;
    state.firing_rate_history = firing_rate_buffer_;
    
    // Neuromodulator levels
    state.neuromodulator_levels = {dopamine_level_, acetylcholine_level_, norepinephrine_level_};
    
    // Learning parameters
    state.learning_rate = learning_rate_;
    state.learning_momentum = 0.9f; // Default momentum
    state.plasticity_threshold = synaptic_tag_threshold_;
    state.homeostatic_target = 0.1f; // Target firing rate
    
    // Performance tracking
    state.learning_steps_count = update_count_;
    state.average_reward = average_activity_; // Use activity as proxy for now
    
    // Copy prediction error history
    for (size_t i = 0; i < std::min(prediction_error_history_.size(), size_t(100)); ++i) {
        state.prediction_error_history[i] = prediction_error_history_[i];
    }
    state.history_index = performance_history_index_;
    
    // Temporal information
    state.last_update_time = std::chrono::system_clock::now();
    
    // Consolidation state
    state.consolidation_strength = consolidation_pending_ ? 1.0f : 0.0f;
    state.needs_consolidation = shouldConsolidate();
    
    // Calculate checksum
    state.calculateChecksum();
    
    return state;
}

std::map<std::string, float> EnhancedNeuralModule::getPerformanceMetrics() const {
    // Placeholder implementation
    return NeuralModule::getPerformanceMetrics();
}

std::vector<float> EnhancedNeuralModule::getOutputs() const {
    // Placeholder implementation
    return NeuralModule::get_output();
}

EnhancedNeuralModule::ModuleState EnhancedNeuralModule::saveState() const {
    // Placeholder implementation
    return ModuleState();
}

void EnhancedNeuralModule::loadState(const EnhancedNeuralModule::ModuleState& state) {
    // Placeholder implementation
}

float EnhancedNeuralModule::getActivityLevel() const {
    return attention_weight_;
}

bool EnhancedNeuralModule::applyLearningState(const ModuleLearningState& state) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Validate state integrity
    if (!state.validateState()) {
        std::cerr << "❌ Invalid learning state for module: " << module_name_ << std::endl;
        return false;
    }
    
    // Validate compatibility
    if (state.module_name != module_name_) {
        std::cerr << "⚠️  Module name mismatch: expected " << module_name_ 
                  << ", got " << state.module_name << std::endl;
    }
    
    try {
        // Apply learning traces
        if (state.eligibility_traces.size() == eligibility_traces_.size()) {
            eligibility_traces_ = state.eligibility_traces;
        } else {
            std::cerr << "⚠️  Eligibility trace size mismatch, resizing..." << std::endl;
            eligibility_traces_.resize(state.eligibility_traces.size());
            eligibility_traces_ = state.eligibility_traces;
        }
        
        if (state.synaptic_tags.size() == synaptic_tags_.size()) {
            synaptic_tags_ = state.synaptic_tags;
        } else {
            synaptic_tags_.resize(state.synaptic_tags.size());
            synaptic_tags_ = state.synaptic_tags;
        }
        
        // Apply neuromodulator levels
        if (state.neuromodulator_levels.size() >= 3) {
            dopamine_level_ = state.neuromodulator_levels[0];
            acetylcholine_level_ = state.neuromodulator_levels[1];
            norepinephrine_level_ = state.neuromodulator_levels[2];
        }
        
        // Apply learning parameters
        learning_rate_ = state.learning_rate;
        synaptic_tag_threshold_ = state.plasticity_threshold;
        
        // Apply performance tracking
        update_count_ = state.learning_steps_count;
        
        // Apply prediction error history
        for (size_t i = 0; i < std::min(prediction_error_history_.size(), size_t(100)); ++i) {
            prediction_error_history_[i] = state.prediction_error_history[i];
        }
        performance_history_index_ = state.history_index;
        
        // Apply consolidation state
        consolidation_pending_ = state.needs_consolidation;
        
        std::cout << "✅ Applied learning state to module: " << module_name_ << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error applying learning state: " << e.what() << std::endl;
        return false;
    }
}

void EnhancedNeuralModule::updateEligibilityTraces(float reward_signal, float dt) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Update eligibility traces for all synapses
    for (size_t i = 0; i < eligibility_traces_.size(); ++i) {
        // Decay existing traces
        eligibility_traces_[i] *= eligibility_decay_rate_;
        
        // Add new trace based on recent activity
        if (i < neuron_outputs_.size() && neuron_outputs_[i] > 0.1f) {
            // Strong activation increases eligibility
            eligibility_traces_[i] += neuron_outputs_[i] * dt;
        }
        
        // Apply reward modulation
        if (std::abs(reward_signal) > 0.01f) {
            // Reward reinforces eligible synapses
            float weight_change = learning_rate_ * reward_signal * eligibility_traces_[i];
            if (i < synaptic_weights_.size()) {
                synaptic_weights_[i] += weight_change;
                
                // Clamp weights to reasonable range
                synaptic_weights_[i] = std::clamp(synaptic_weights_[i], -1.0f, 1.0f);
            }
        }
        
        // Clamp eligibility traces
        eligibility_traces_[i] = std::clamp(eligibility_traces_[i], 0.0f, 1.0f);
    }
}

void EnhancedNeuralModule::applySynapticTagging(float novelty_signal) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    if (std::abs(novelty_signal) < 0.01f) {
        return; // No significant novelty
    }
    
    // Apply synaptic tagging based on current activity and novelty
    for (size_t i = 0; i < synaptic_tags_.size(); ++i) {
        // Decay existing tags
        synaptic_tags_[i] *= 0.99f;
        
        // Strong activity + novelty creates tags
        if (i < neuron_outputs_.size() && neuron_outputs_[i] > synaptic_tag_threshold_) {
            float tag_strength = neuron_outputs_[i] * std::abs(novelty_signal);
            synaptic_tags_[i] = std::max(synaptic_tags_[i], tag_strength);
        }
        
        // Clamp tags
        synaptic_tags_[i] = std::clamp(synaptic_tags_[i], 0.0f, 1.0f);
    }
}

void EnhancedNeuralModule::updateNeuromodulators(float dopamine, float acetylcholine, float norepinephrine) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    // Update neuromodulator levels with decay
    dopamine_level_ = 0.9f * dopamine_level_ + 0.1f * dopamine;
    acetylcholine_level_ = 0.95f * acetylcholine_level_ + 0.05f * acetylcholine;
    norepinephrine_level_ = 0.8f * norepinephrine_level_ + 0.2f * norepinephrine;
    
    // Clamp to physiological ranges
    dopamine_level_ = std::clamp(dopamine_level_, 0.0f, 2.0f);
    acetylcholine_level_ = std::clamp(acetylcholine_level_, 0.0f, 1.0f);
    norepinephrine_level_ = std::clamp(norepinephrine_level_, 0.0f, 1.5f);
    
    // Modulate learning rate based on neuromodulators
    float modulation_factor = 1.0f + 0.5f * dopamine_level_ + 0.3f * acetylcholine_level_;
    // Don't directly modify learning_rate_ here to maintain base rate
}

size_t EnhancedNeuralModule::performMemoryConsolidation(float consolidation_strength) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    size_t consolidated_synapses = 0;
    
    for (size_t i = 0; i < synaptic_weights_.size(); ++i) {
        // Consolidate synapses with high eligibility and tags
        float consolidation_score = 0.0f;
        
        if (i < eligibility_traces_.size()) {
            consolidation_score += eligibility_traces_[i];
        }
        if (i < synaptic_tags_.size()) {
            consolidation_score += synaptic_tags_[i];
        }
        
        if (consolidation_score > consolidation_threshold_) {
            // Strengthen the synapse
            float strengthening = consolidation_strength * consolidation_score;
            synaptic_weights_[i] += strengthening * synaptic_weights_[i];
            
            // Decay the traces/tags after consolidation
            if (i < eligibility_traces_.size()) {
                eligibility_traces_[i] *= 0.8f;
            }
            if (i < synaptic_tags_.size()) {
                synaptic_tags_[i] *= 0.8f;
            }
            
            consolidated_synapses++;
        }
    }
    
    // Update consolidation timing
    last_consolidation_ = std::chrono::steady_clock::now();
    consolidation_pending_ = false;
    
    std::cout << "🧠 Consolidated " << consolidated_synapses << " synapses in module: " 
              << module_name_ << std::endl;
    
    return consolidated_synapses;
}

std::vector<float> EnhancedNeuralModule::getSynapticWeights() const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    return synaptic_weights_;
}

bool EnhancedNeuralModule::setSynapticWeights(const std::vector<float>& weights) {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    if (weights.size() != synaptic_weights_.size()) {
        std::cerr << "❌ Weight vector size mismatch: expected " << synaptic_weights_.size() 
                  << ", got " << weights.size() << std::endl;
        return false;
    }
    
    synaptic_weights_ = weights;
    return true;
}

float EnhancedNeuralModule::getPredictionError(const std::vector<float>& expected_output) const {
    std::lock_guard<std::mutex> lock(module_mutex_);
    
    if (expected_output.size() != neuron_outputs_.size()) {
        return 1.0f; // Maximum error for size mismatch
    }
    
    float total_error = 0.0f;
    for (size_t i = 0; i < neuron_outputs_.size(); ++i) {
        float error = expected_output[i] - neuron_outputs_[i];
        total_error += error * error; // MSE
    }
    
    return std::sqrt(total_error / neuron_outputs_.size());
}

void EnhancedNeuralModule::updatePerformanceHistory(float prediction_error) {
    prediction_error_history_[performance_history_index_] = prediction_error;
    performance_history_index_ = (performance_history_index_ + 1) % prediction_error_history_.size();
}

bool EnhancedNeuralModule::shouldConsolidate() const {
    auto now = std::chrono::steady_clock::now();
    auto minutes_since_last = std::chrono::duration_cast<std::chrono::minutes>(now - last_consolidation_).count();
    
    // Consolidate if it's been more than 30 minutes or if there's high learning activity
    if (minutes_since_last > 30) {
        return true;
    }
    
    // Check if there's significant tagged activity
    float total_tags = 0.0f;
    for (float tag : synaptic_tags_) {
        total_tags += tag;
    }
    
    return (total_tags / synaptic_tags_.size()) > 0.1f;
}

void EnhancedNeuralModule::applyNeuromodulation(const std::string& modulator_type, float level) {
    // Placeholder implementation
    std::lock_guard<std::mutex> lock(module_mutex_);
    neuromodulator_levels_[modulator_type] = level;
}

// ============================================================================
// ENHANCED UPDATE METHOD
// ============================================================================

void EnhancedNeuralModule::update(float dt, const std::vector<float>& inputs, float reward) {
    // Call parent update
    NeuralModule::update(dt, inputs, reward);
    
    // Update learning traces
    updateEligibilityTraces(reward, dt);
    
    // Apply synaptic tagging based on novelty (simplified)
    float novelty_signal = std::abs(reward); // Use reward as novelty proxy
    applySynapticTagging(novelty_signal);
    
    // Update neuromodulators based on reward and context
    float dopamine = reward; // Reward prediction error
    float acetylcholine = getActivityLevel(); // Attention based on activity
    float norepinephrine = std::abs(reward) > 0.5f ? 1.0f : 0.0f; // Arousal
    
    updateNeuromodulators(dopamine, acetylcholine, norepinephrine);
    
    // Update performance history
    if (!inputs.empty()) {
        std::vector<float> expected_output(neuron_outputs_.size(), 0.5f); // Simple expectation
        float prediction_error = getPredictionError(expected_output);
        updatePerformanceHistory(prediction_error);
    }
    
    // Check if consolidation is needed
    if (shouldConsolidate()) {
        consolidation_pending_ = true;
    }
}

// ============================================================================
// CHECKSUM AND VALIDATION IMPLEMENTATION
// ============================================================================

void ModuleLearningState::calculateChecksum() {
    // Simple checksum based on key values
    std::hash<float> float_hasher;
    std::hash<std::string> string_hasher;
    
    checksum = 0;
    checksum ^= string_hasher(module_name);
    checksum ^= static_cast<uint64_t>(learning_rate * 1000000);
    checksum ^= learning_steps_count;
    
    // Add contribution from traces
    for (size_t i = 0; i < std::min(eligibility_traces.size(), size_t(100)); ++i) {
        checksum ^= float_hasher(eligibility_traces[i]) + i;
    }
}

bool ModuleLearningState::validateState() const {
    // Check version compatibility
    if (state_version != 1) {
        return false;
    }
    
    // Check reasonable ranges
    if (learning_rate < 0.0f || learning_rate > 1.0f) {
        return false;
    }
    
    if (plasticity_threshold < 0.0f || plasticity_threshold > 1.0f) {
        return false;
    }
    
    // Check vector sizes are reasonable
    if (eligibility_traces.size() > 1000000 || synaptic_tags.size() > 1000000) {
        return false;
    }
    
    // Validate checksum
    ModuleLearningState temp_state = *this;
    temp_state.calculateChecksum();
    
    return temp_state.checksum == checksum;
}