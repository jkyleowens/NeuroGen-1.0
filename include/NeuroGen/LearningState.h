// Update 1: Enhanced Learning State Structures
// File: include/NeuroGen/LearningState.h (NEW FILE)

#ifndef LEARNING_STATE_H
#define LEARNING_STATE_H

#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <cstdint>

/**
 * @brief Extended learning state for persistent neural modules
 */
struct ModuleLearningState {
    // Basic module identification
    std::string module_name;
    uint32_t module_id;
    
    // Learning-specific state vectors
    std::vector<float> eligibility_traces;      // Per-synapse eligibility traces
    std::vector<float> synaptic_tags;           // Late-phase LTP markers
    std::vector<float> neuromodulator_levels;   // Dopamine, acetylcholine, etc.
    std::vector<float> firing_rate_history;    // Recent firing rate buffer
    
    // Learning parameters
    float learning_rate;
    float learning_momentum;
    float plasticity_threshold;
    float homeostatic_target;
    
    // Performance tracking
    uint64_t learning_steps_count;
    float average_reward;
    float prediction_error_history[100];       // Circular buffer
    uint32_t history_index;
    
    // Temporal information
    std::chrono::system_clock::time_point last_update_time;
    std::chrono::system_clock::time_point creation_time;
    
    // Consolidation state
    float consolidation_strength;
    bool needs_consolidation;
    uint32_t consolidation_count;
    
    // Validation and versioning
    uint32_t state_version;
    uint64_t checksum;
    
    ModuleLearningState() 
        : module_id(0), learning_rate(0.001f), learning_momentum(0.9f),
          plasticity_threshold(0.1f), homeostatic_target(0.1f),
          learning_steps_count(0), average_reward(0.0f),
          history_index(0), consolidation_strength(0.0f),
          needs_consolidation(false), consolidation_count(0),
          state_version(1), checksum(0) {
        
        std::fill(std::begin(prediction_error_history), 
                  std::end(prediction_error_history), 0.0f);
        creation_time = std::chrono::system_clock::now();
        last_update_time = creation_time;
    }
    
    /**
     * @brief Calculate checksum for state validation
     */
    void calculateChecksum();
    
    /**
     * @brief Validate state integrity
     */
    bool validateState() const;
};

/**
 * @brief Inter-module connection learning state
 */
struct InterModuleConnectionState {
    std::string source_module;
    std::string target_module;
    std::string connection_type;  // "excitatory", "inhibitory", "modulatory"
    
    float connection_strength;
    float base_strength;          // Initial strength for recovery
    float plasticity_rate;
    float usage_frequency;
    
    // Hebbian learning traces
    std::vector<float> pre_synaptic_trace;
    std::vector<float> post_synaptic_trace;
    float correlation_strength;
    
    // Timing-dependent plasticity
    std::vector<float> spike_timing_buffer;
    uint32_t timing_buffer_index;
    
    // Performance metrics
    float information_transfer_rate;
    float mutual_information;
    uint64_t activation_count;
    
    std::chrono::system_clock::time_point last_activation;
    
    InterModuleConnectionState()
        : connection_strength(0.5f), base_strength(0.5f),
          plasticity_rate(0.001f), usage_frequency(0.0f),
          correlation_strength(0.0f), timing_buffer_index(0),
          information_transfer_rate(0.0f), mutual_information(0.0f),
          activation_count(0) {
        
        pre_synaptic_trace.resize(1000, 0.0f);  // 1 second at 1kHz
        post_synaptic_trace.resize(1000, 0.0f);
        spike_timing_buffer.resize(100, 0.0f);
        last_activation = std::chrono::system_clock::now();
    }
};

/**
 * @brief Global learning session state
 */
struct SessionLearningState {
    std::string session_id;
    std::string session_name;
    uint64_t session_number;
    
    std::chrono::system_clock::time_point session_start;
    std::chrono::system_clock::time_point last_checkpoint;
    
    // Global learning parameters
    float global_learning_rate_multiplier;
    float exploration_rate;
    float curiosity_drive;
    
    // Experience and performance
    uint64_t total_experiences;
    uint64_t total_learning_steps;
    float cumulative_reward;
    float average_performance;
    
    // Memory consolidation state
    uint32_t consolidation_cycles;
    std::chrono::system_clock::time_point last_consolidation;
    
    // Module states
    std::map<std::string, ModuleLearningState> module_states;
    std::vector<InterModuleConnectionState> inter_module_connections;
    
    // Architecture compatibility
    std::string architecture_hash;
    uint32_t total_modules;
    uint32_t total_neurons;
    uint32_t total_synapses;
    
    SessionLearningState()
        : session_number(0), global_learning_rate_multiplier(1.0f),
          exploration_rate(0.3f), curiosity_drive(0.5f),
          total_experiences(0), total_learning_steps(0),
          cumulative_reward(0.0f), average_performance(0.0f),
          consolidation_cycles(0), total_modules(0),
          total_neurons(0), total_synapses(0) {
        
        session_start = std::chrono::system_clock::now();
        last_checkpoint = session_start;
        last_consolidation = session_start;
    }
};

#endif // LEARNING_STATE_H