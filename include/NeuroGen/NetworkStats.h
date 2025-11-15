#ifndef NETWORK_STATS_H
#define NETWORK_STATS_H

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>

// ============================================================================
// PLATFORM-SPECIFIC DEFINITIONS
// ============================================================================

#ifdef __CUDACC__
    #define STATS_HOST_DEVICE __host__ __device__
    #define STATS_MANAGED __managed__
#else
    #define STATS_HOST_DEVICE
    #define STATS_MANAGED
#endif

/**
 * @class NetworkStats
 * @brief Comprehensive statistics collection for neural network simulations.
 *
 * This class provides a unified interface for collecting, storing, and analyzing
 * performance metrics from neural network simulations. It tracks everything from
 * basic activity metrics to advanced computational performance indicators.
 * Optimized for real-time brain simulation and supports both CPU and CUDA backends.
 */
class NetworkStats {
public:
    // ========================================================================
    // CORE SIMULATION METRICS
    // ========================================================================
    
    /** Current simulation time in milliseconds */
    float current_time_ms = 0.0f;
    
    /** Total elapsed simulation time */
    float total_simulation_time = 0.0f;
    
    /** Current reward signal for learning algorithms */
    float current_reward = 0.0f;
    
    /** Cumulative reward over simulation */
    float cumulative_reward = 0.0f;
    
    /** Average reward over recent time window */
    float average_reward = 0.0f;
    
    /** Number of simulation steps completed */
    uint64_t simulation_steps = 0;
    
    // ========================================================================
    // NETWORK ACTIVITY METRICS
    // ========================================================================
    
    /** Total number of spikes generated this timestep */
    uint32_t current_spike_count = 0;
    
    /** Total spikes across entire simulation */
    uint64_t total_spike_count = 0;
    
    /** **FIXED: Removed reference alias to fix copy assignment operator** */
    
    /** Mean firing rate across all neurons (Hz) */
    float mean_firing_rate = 0.0f;
    
    /** Standard deviation of firing rates */
    float firing_rate_std = 0.0f;
    
    /** Network-wide synchrony measure (0-1) */
    float network_synchrony = 0.0f;
    
    /** Percentage of neurons active this timestep */
    float neuron_activity_ratio = 0.0f;
    
    /** Number of neurons that spiked */
    uint32_t active_neuron_count = 0;
    
    /** Total number of neurons in network */
    uint32_t total_neurons = 0;
    
    /** Population vector magnitude (activity coherence) */
    float population_vector_strength = 0.0f;
    
    // ========================================================================
    // SYNAPTIC AND CONNECTIVITY METRICS  
    // ========================================================================
    
    /** Total number of synapses in network */
    uint32_t total_synapses = 0;
    
    /** Number of active synapses (weight > threshold) */
    uint32_t active_synapses = 0;
    
    /** Mean synaptic weight */
    float mean_synaptic_weight = 0.0f;
    
    /** Standard deviation of synaptic weights */
    float synaptic_weight_std = 0.0f;
    
    /** Rate of synaptic weight change (plasticity) */
    float plasticity_rate = 0.0f;
    
    /** Connection density (synapses/possible connections) */
    float connection_density = 0.0f;
    
    /** Small-world connectivity measure */
    float small_world_coefficient = 0.0f;
    
    /** Network clustering coefficient */
    float clustering_coefficient = 0.0f;
    
    /** Average path length between neurons */
    float average_path_length = 0.0f;
    
    // ========================================================================
    // NEUROMODULATION METRICS
    // ========================================================================
    
    /** Dopamine concentration level */
    float dopamine_level = 0.0f;
    
    /** Acetylcholine concentration level */
    float acetylcholine_level = 0.0f;
    
    /** Serotonin concentration level */
    float serotonin_level = 0.0f;
    
    /** Norepinephrine concentration level */
    float norepinephrine_level = 0.0f;
    
    /** GABA concentration level */
    float gaba_level = 0.0f;
    
    /** Glutamate concentration level */
    float glutamate_level = 0.0f;
    
    /** Overall neuromodulatory balance */
    float neuromod_balance = 0.0f;
    
    /** Excitation/inhibition ratio */
    float excitation_inhibition_ratio = 1.0f;
    
    // ========================================================================
    // COMPUTATIONAL PERFORMANCE METRICS
    // ========================================================================
    
    /** Time spent in CUDA kernels (ms) */
    float kernel_execution_time = 0.0f;
    
    /** Memory transfer time (ms) */
    float memory_transfer_time = 0.0f;
    
    /** Total memory usage (bytes) */
    uint64_t memory_usage_bytes = 0;
    
    /** GPU utilization percentage */
    float gpu_utilization = 0.0f;
    
    /** Simulation speed factor (real-time = 1.0) */
    float simulation_speed_factor = 1.0f;
    
    /** Energy consumption estimate (arbitrary units) */
    float energy_consumption = 0.0f;
    
    /** Number of CUDA errors encountered */
    uint32_t cuda_error_count = 0;
    
    /** Whether simulation is numerically stable */
    bool simulation_stable = true;
    
    /** Numerical stability score (0-1) */
    float numerical_stability = 1.0f;
    
    // ========================================================================
    // LEARNING AND ADAPTATION METRICS
    // ========================================================================
    
    /** Rate of learning (weight change magnitude) */
    float learning_rate = 0.0f;
    
    /** Prediction error (for supervised learning) */
    float prediction_error = 0.0f;
    
    /** Information capacity estimate */
    float information_capacity = 0.0f;
    
    /** Network entropy measure */
    float network_entropy = 0.0f;
    
    /** Mutual information between layers */
    float mutual_information = 0.0f;
    
    /** Adaptation rate to input changes */
    float adaptation_rate = 0.0f;
    
    // ========================================================================
    // COLUMNAR AND MODULAR ORGANIZATION
    // ========================================================================
    
    /** Activity per cortical column */
    std::vector<float> column_activity;
    
    /** Frequency bands per column */
    std::vector<float> column_frequencies;
    
    /** Specialization index per column */
    std::vector<float> column_specialization;
    
    /** Performance tracking history */
    std::vector<float> performance_history;
    
    // ========================================================================
    // METHODS FOR UPDATING STATISTICS
    // ========================================================================
    
    /** Update basic simulation metrics */
    STATS_HOST_DEVICE void updateSimulationMetrics(float dt, uint64_t steps) {
        current_time_ms += dt * 1000.0f;
        total_simulation_time += dt * 1000.0f;
        simulation_steps += steps;
    }
    
    /** Update neural activity metrics */
    STATS_HOST_DEVICE void updateActivityMetrics(uint32_t spikes, uint32_t active, uint32_t total) {
        current_spike_count = spikes;
        total_spike_count += spikes;
        active_neuron_count = active;
        total_neurons = total;
        neuron_activity_ratio = (total > 0) ? static_cast<float>(active) / total : 0.0f;
    }
    
    /** Update synaptic metrics */
    STATS_HOST_DEVICE void updateSynapticMetrics(float mean_weight, float weight_std, 
                                               uint32_t active_syn, uint32_t total_syn) {
        mean_synaptic_weight = mean_weight;
        synaptic_weight_std = weight_std;
        active_synapses = active_syn;
        total_synapses = total_syn;
        connection_density = (total_syn > 0) ? static_cast<float>(active_syn) / total_syn : 0.0f;
    }
    
    /** Update neuromodulation levels */
    STATS_HOST_DEVICE void updateNeuromodulation(float da, float ach, float ser, float nor) {
        dopamine_level = da;
        acetylcholine_level = ach;
        serotonin_level = ser;
        norepinephrine_level = nor;
        
        // Calculate balance metric
        float total = da + ach + ser + nor;
        neuromod_balance = (total > 0.0f) ? (4.0f * 0.25f) / total : 0.0f;
    }
    
    /** Update performance metrics */
    STATS_HOST_DEVICE void updatePerformanceMetrics(float kernel_time, float transfer_time, 
                                                   uint64_t memory_bytes) {
        kernel_execution_time = kernel_time;
        memory_transfer_time = transfer_time;
        memory_usage_bytes = memory_bytes;
        
        // Calculate simulation speed (inverse of total time)
        float total_time = kernel_time + transfer_time;
        simulation_speed_factor = (total_time > 0.0f) ? 1.0f / total_time : 1.0f;
    }
    
    /** Reset all statistics to default values */
    STATS_HOST_DEVICE void reset() {
        *this = NetworkStats();
    }
    
    // ========================================================================
    // ANALYSIS AND SCORING METHODS
    // ========================================================================
    
    /** Get overall network health score (0-1) */
    STATS_HOST_DEVICE float getHealthScore() const {
        float activity_score = (neuron_activity_ratio > 0.1f && neuron_activity_ratio < 0.9f) ? 
                               1.0f : 0.5f;
        float plasticity_score = (plasticity_rate > 0.0f && plasticity_rate < 0.1f) ? 1.0f : 0.5f;
        float stability_score = numerical_stability;
        
        return (activity_score + plasticity_score + stability_score) / 3.0f;
    }
    
    /** Get computational performance score */
    STATS_HOST_DEVICE float getPerformanceScore() const {
        return (simulation_speed_factor > 1.0f) ? 1.0f : simulation_speed_factor;
    }
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /** Convert statistics to human-readable string */
    std::string toString() const;
    
    /** Export detailed statistics to JSON format */
    std::string toJSON() const;
    
    /** Save statistics to binary file */
    bool saveToFile(const std::string& filename) const;
    
    /** Load statistics from binary file */
    bool loadFromFile(const std::string& filename);
    
    // ========================================================================
    // BACKWARD COMPATIBILITY
    // ========================================================================
    
    /** **FIXED: Added getter for backward compatibility with total_spikes** */
    STATS_HOST_DEVICE uint64_t getTotalSpikes() const {
        return total_spike_count;
    }
    
    /** **FIXED: Added setter for backward compatibility with total_spikes** */
    STATS_HOST_DEVICE void setTotalSpikes(uint64_t spikes) {
        total_spike_count = spikes;
    }
};

// ============================================================================
// GLOBAL STATISTICS INSTANCES
// ============================================================================

#ifdef __CUDACC__
/** Global managed statistics for CUDA kernels */
extern STATS_MANAGED NetworkStats g_stats;
#else
/** Global statistics instance for CPU code */
extern NetworkStats g_stats;
#endif

// ============================================================================
// CONVENIENCE MACROS FOR STATISTICS UPDATES
// ============================================================================

#define UPDATE_SIMULATION_STATS(dt, steps) \
    g_stats.updateSimulationMetrics(dt, steps)

#define UPDATE_ACTIVITY_STATS(spikes, active, total) \
    g_stats.updateActivityMetrics(spikes, active, total)

#define UPDATE_SYNAPTIC_STATS(mean_w, std_w, active_s, total_s) \
    g_stats.updateSynapticMetrics(mean_w, std_w, active_s, total_s)

#define UPDATE_NEUROMOD_STATS(da, ach, ser, nor) \
    g_stats.updateNeuromodulation(da, ach, ser, nor)

#define UPDATE_PERFORMANCE_STATS(kernel_t, transfer_t, mem_bytes) \
    g_stats.updatePerformanceMetrics(kernel_t, transfer_t, mem_bytes)

#define STATS_RECORD_ERROR() \
    do { g_stats.cuda_error_count++; g_stats.simulation_stable = false; } while(0)

#endif // NETWORK_STATS_H