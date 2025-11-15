// ============================================================================
// SPECIALIZED MODULE IMPLEMENTATION
// File: src/SpecializedModule.cpp
// ============================================================================

#include "NeuroGen/SpecializedModule.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

SpecializedModule::SpecializedModule(const std::string& name, const NetworkConfig& config, 
                                   const std::string& module_type)
    : EnhancedNeuralModule(name, config),
      specialization_type_(module_type),
      attention_weight_(1.0f),
      activation_threshold_(0.5f),
      learning_rate_(0.01f),
      decay_rate_(0.95f),
      noise_level_(0.01f) {
    
    // Initialize default state and buffer sizes based on config
    size_t state_size = config.num_neurons ? config.num_neurons : 100;
    size_t output_size = state_size / 2; // Output is typically smaller than internal state
    
    initialize_internal_state(state_size, output_size);
    
    std::cout << "SpecializedModule: Created " << specialization_type_ 
              << " module '" << name << "' with " << state_size 
              << " state units and " << output_size << " outputs." << std::endl;
}

bool SpecializedModule::initialize() {
    // Call parent initialization first
    if (!EnhancedNeuralModule::initialize()) {
        std::cerr << "SpecializedModule: Failed to initialize parent class" << std::endl;
        return false;
    }
    
    // Specialized initialization based on module type
    if (specialization_type_ == "motor") {
        activation_threshold_ = 0.3f;  // Lower threshold for motor responses
        learning_rate_ = 0.02f;        // Higher learning rate for motor adaptation
        noise_level_ = 0.02f;          // More noise for motor variability
    } else if (specialization_type_ == "attention") {
        activation_threshold_ = 0.7f;  // Higher threshold for attention focus
        learning_rate_ = 0.05f;        // Fast attention updates
        decay_rate_ = 0.8f;            // Faster decay for dynamic attention
    } else if (specialization_type_ == "reward") {
        activation_threshold_ = 0.1f;  // Sensitive to small reward signals
        learning_rate_ = 0.01f;        // Slow, stable learning for value estimation
        decay_rate_ = 0.98f;           // Slow decay for reward memory
    } else if (specialization_type_ == "memory") {
        activation_threshold_ = 0.4f;  // Moderate threshold for memory gating
        learning_rate_ = 0.008f;       // Slow learning for stable memory
        decay_rate_ = 0.99f;           // Very slow decay for memory persistence
    }
    
    std::cout << "SpecializedModule: Initialized " << specialization_type_ 
              << " module with threshold=" << activation_threshold_ 
              << ", learning_rate=" << learning_rate_ << std::endl;
    
    return true;
}

// ========================================================================
// MODULE CONFIGURATION AND CONTROL
// ========================================================================

void SpecializedModule::set_specialization_type(const std::string& type) {
    specialization_type_ = type;
    // Re-initialize with new type-specific parameters
    initialize();
}

const std::string& SpecializedModule::get_specialization_type() const {
    return specialization_type_;
}

void SpecializedModule::set_attention_weight(float weight) {
    attention_weight_ = std::max(0.0f, std::min(1.0f, weight));
}

float SpecializedModule::get_attention_weight() const {
    return attention_weight_;
}

void SpecializedModule::set_activation_threshold(float threshold) {
    activation_threshold_ = threshold;
}

float SpecializedModule::get_activation_threshold() const {
    return activation_threshold_;
}

// ========================================================================
// OVERRIDDEN VIRTUAL FUNCTIONS
// ========================================================================

void SpecializedModule::update(float dt, const std::vector<float>& inputs, float reward) {
    // Call parent update first
    EnhancedNeuralModule::update(dt, inputs, reward);
    
    // Apply specialized processing based on module type
    if (!inputs.empty()) {
        if (specialization_type_ == "motor") {
            output_buffer_ = process_motor_cortex(inputs);
        } else if (specialization_type_ == "attention") {
            output_buffer_ = process_attention_system(inputs);
        } else if (specialization_type_ == "reward") {
            output_buffer_ = process_reward_system(inputs);
        } else if (specialization_type_ == "memory") {
            output_buffer_ = process_working_memory(inputs);
        } else {
            // General processing - just copy inputs with some processing
            output_buffer_.resize(std::min(inputs.size(), output_buffer_.size()));
            for (size_t i = 0; i < output_buffer_.size(); ++i) {
                float processed = std::tanh(inputs[i] * attention_weight_ - activation_threshold_);
                output_buffer_[i] = apply_noise(processed);
            }
        }
    }
    
    // Update internal metrics
    update_internal_metrics();
}

std::vector<float> SpecializedModule::get_output() const {
    return output_buffer_;
}

// ========================================================================
// HELPER METHODS
// ========================================================================

void SpecializedModule::initialize_internal_state(size_t state_size, size_t output_size) {
    internal_state_.resize(state_size, 0.0f);
    output_buffer_.resize(output_size, 0.0f);
    
    // Initialize with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (float& state : internal_state_) {
        state = dist(gen);
    }
}

float SpecializedModule::apply_noise(float signal) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(0.0f, noise_level_);
    
    return signal + noise_dist(gen);
}

void SpecializedModule::update_internal_metrics() {
    // Calculate average activity for performance monitoring
    float total_activity = 0.0f;
    for (const float& state : internal_state_) {
        total_activity += std::abs(state);
    }

    if (!internal_state_.empty()) {
        // Update performance metrics that can be accessed by the controller
        // float avg_activity = total_activity / internal_state_.size();

        // This would typically update performance metrics in the parent class
        // The exact implementation depends on the parent class interface
    }
}

void SpecializedModule::apply_reinforcement(float reward, float global_reward) {
    // This is a placeholder implementation.
    // The actual implementation would involve more complex logic, 
    // such as updating weights based on the reward signal.
    
    // std::cout << "Applying reinforcement to module " << get_name() 
    //           << " with reward " << reward 
    //           << " and global reward " << global_reward << std::endl;

    // Example of a simple learning rule - update internal state based on reward
    for (float& state : internal_state_) {
        state += learning_rate_ * reward * (state > 0 ? 1.0f : -1.0f);
    }
}
