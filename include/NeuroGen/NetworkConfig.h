#ifndef NEUROGENALPHA_NETWORKCONFIG_H
#define NEUROGENALPHA_NETWORKCONFIG_H

#include <iostream>
#include <string>
#include <cstddef>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#endif

/**
 * @brief Network configuration parameters
 */
struct NetworkConfig {
    // Simulation parameters
    double dt = 0.01;                    // Integration time step (ms)
    double axonal_speed = 1.0;           // m/s (affects delays)
    
    // Spatial organization
    double network_width = 1000.0;       // μm
    double network_height = 1000.0;      // μm
    double network_depth = 100.0;        // μm
    
    bool enable_structural_plasticity = false;
    size_t num_neurons = 0; // The actual number of neurons to create.

    // Connectivity parameters
    double max_connection_distance = 200.0; // μm
    double connection_probability_base = 0.01;
    double distance_decay_constant = 50.0;   // μm
    double spike_correlation_window = 20.0;  // ms
    double correlation_threshold = 0.3;
    
    // Neurogenesis parameters
    bool enable_neurogenesis = true;
    double neurogenesis_rate = 0.001;        // neurons/ms base rate
    double activity_threshold_low = 0.1;     // For underactivation
    double activity_threshold_high = 10.0;   // For hyperactivation
    size_t max_neurons = 65536;              // MASSIVE SCALE-UP: Support up to 64K neurons per module
    
    // Pruning parameters
    bool enable_pruning = true;
    double synapse_pruning_threshold = 0.05;
    double neuron_pruning_threshold = 0.01;
    double pruning_check_interval = 100.0;   // ms
    double synapse_activity_window = 1000.0; // ms
    
    // Plasticity parameters
    bool enable_stdp = true;
    double stdp_learning_rate = 0.01;
    double stdp_tau_pre = 20.0;              // ms
    double stdp_tau_post = 20.0;             // ms
    double eligibility_decay = 50.0;         // ms
    double min_synaptic_weight = 0.001;      // Minimum synaptic weight
    double max_synaptic_weight = 2.0;        // Maximum synaptic weight
    
    // STDP parameters (additional for CUDA compatibility)
    float reward_learning_rate = 0.01f;      // Reward modulation learning rate
    float A_plus = 0.01f;                    // STDP potentiation amplitude
    float A_minus = 0.012f;                  // STDP depression amplitude
    float tau_plus = 20.0f;                  // STDP potentiation time constant (ms)
    float tau_minus = 20.0f;                 // STDP depression time constant (ms)
    float min_weight = 0.001f;               // Minimum synaptic weight (alias for CUDA)
    float max_weight = 2.0f;                 // Maximum synaptic weight (alias for CUDA)
    
    // Homeostatic parameters
    float homeostatic_strength = 0.001f;     // Homeostatic scaling strength
    
    // Network topology parameters
    int input_size = 256;                    // INCREASED: 256 input neurons for richer input processing
    int output_size = 64;                    // INCREASED: 64 output neurons for complex action space  
    int hidden_size = 8192;                  // MASSIVE SCALE-UP: 8K hidden neurons for emergent intelligence
    
    // Connection probabilities - OPTIMIZED for large-scale networks
    float input_hidden_prob = 0.15f;         // REDUCED: Sparse connectivity for computational efficiency
    float hidden_hidden_prob = 0.05f;        // REDUCED: Very sparse recurrent connections for stability  
    float hidden_output_prob = 0.3f;         // REDUCED: Selective output connections for focused behavior
    
    // Weight initialization
    float weight_init_std = 0.5f;            // Standard deviation for weight initialization
    float delay_min = 1.0f;                  // Minimum synaptic delay (ms)
    float delay_max = 5.0f;                  // Maximum synaptic delay (ms)
    
    // Input parameters
    float input_current_scale = 10.0f;       // Scale factor for input current injection    

    float exc_ratio = 0.8f;  // Excitatory connection ratio
    float simulation_time = 50.0f;

    // TopologyGenerator-specific fields - SCALED UP for massive neural architecture
    int numColumns = 16;                      // INCREASED: 16 cortical columns for complex organization
    int neuronsPerColumn = 512;               // MASSIVE INCREASE: 512 neurons per column (16×512 = 8,192 total)
    int localFanOut = 50;                     // INCREASED: 50 local fan-out connections for richer dynamics
    int localFanIn = 50;                      // INCREASED: 50 local fan-in connections for complex integration
    
    // Synaptic weight ranges
    float wExcMin = 0.05f;                    // Minimum excitatory weight
    float wExcMax = 0.15f;                    // Maximum excitatory weight
    float wInhMin = 0.20f;                    // Minimum inhibitory weight  
    float wInhMax = 0.40f;                    // Maximum inhibitory weight
    
    // Synaptic delay ranges
    float dMin = 0.5f;                        // Minimum synaptic delay (ms)
    float dMax = 2.0f;                        // Maximum synaptic delay (ms)
    
    // Computed fields
    size_t totalSynapses = 0;                 // Total synapses (computed by finalizeConfig)

    bool enable_monitoring = true;
    int monitoring_interval = 100;
    
    // Neuromodulation
    bool enable_neuromodulation = true;
    double modulation_strength = 0.1;

    // Spike threshold
    double spike_threshold = 30.0;           // mV
    
    //Pruning threshold
    float pruning_threshold = 0.1f;

    
    void print() const {
        std::cout << "=== Network Configuration ===" << std::endl;
        std::cout << "Input Size: " << input_size << std::endl;
        std::cout << "Hidden Size: " << hidden_size << std::endl;
        std::cout << "Output Size: " << output_size << std::endl;
        std::cout << "Simulation Time: " << simulation_time << " ms" << std::endl;
        std::cout << "Time Step: " << dt << " ms" << std::endl;
        std::cout << "Excitatory Ratio: " << exc_ratio << std::endl;
        std::cout << "============================" << std::endl;
    }
    
    // Validation method
    bool validate() const {
        // >>> FIX: Added '&&' to create a valid boolean expression
        return input_size > 0 && output_size > 0 && hidden_size > 0 &&
               min_weight >= 0.0f && max_weight > min_weight &&
               tau_plus > 0.0f && tau_minus > 0.0f &&
               A_plus >= 0.0f && A_minus >= 0.0f &&
               numColumns > 0 && neuronsPerColumn > 0 &&
               localFanOut > 0 && wExcMin >= 0.0f && wExcMax > wExcMin &&
               wInhMin >= 0.0f && wInhMax > wInhMin &&
               dMin > 0.0f && dMax > dMin;
    }
    
    void finalizeConfig() {
        totalSynapses = static_cast<size_t>(numColumns) * static_cast<size_t>(neuronsPerColumn) * static_cast<size_t>(localFanOut);
        if (wExcMax <= wExcMin) { wExcMax = wExcMin + 0.1f; }
        if (wInhMax <= wInhMin) { wInhMax = wInhMin + 0.1f; }
        if (dMax <= dMin) { dMax = dMin + 0.5f; }
        hidden_size = numColumns * neuronsPerColumn;
        if (dt <= 0.0) { dt = 0.01; }
    }
    
    std::string toString() const {
        return "NetworkConfig{dt=" + std::to_string(dt) +
               ", max_neurons=" + std::to_string(hidden_size) + 
               ", numColumns=" + std::to_string(numColumns) +
               ", neuronsPerColumn=" + std::to_string(neuronsPerColumn) + "}";
    }
};

#endif // NEUROGENALPHA_NETWORKCONFIG_H