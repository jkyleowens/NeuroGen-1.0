#ifndef NETWORK_H
#define NETWORK_H

// ============================================================================
// CORE SYSTEM INCLUDES - Optimized for Neural Computation
// ============================================================================
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <fstream>

// ============================================================================
// NEUREGEN FRAMEWORK INCLUDES - Modular Architecture Components
// ============================================================================
#include <NeuroGen/Neuron.h>
#include <NeuroGen/Synapse.h>
#include <NeuroGen/NetworkStats.h>
#include <NeuroGen/NetworkConfig.h>

// ============================================================================
// FORWARD DECLARATIONS - Prevent Circular Dependencies
// ============================================================================
class NeuralModule;
class NetworkBuilder;
class GPUNeuralManager;

/**
 * @class Network
 * @brief Advanced modular neural network implementing breakthrough brain-inspired architecture
 *
 * This class represents the core of a biologically-realistic neural simulation system
 * designed to mirror cortical column organization with independent, self-contained modules.
 * Features include:
 * - Variable-size modular architecture with hierarchical processing
 * - Independent state saving/loading for each network module  
 * - Advanced synaptic plasticity with STDP, homeostasis, and reward modulation
 * - Structural plasticity enabling dynamic synaptic pruning and growth
 * - Central attention/control mechanisms for context-dependent orchestration
 * - Biological realism with Dale's principle and realistic firing dynamics
 * 
 * The network supports the development of truly brain-like computation through
 * sophisticated interconnection of specialized modules that adapt their function
 * based on input patterns and contextual requirements.
 */
class Network {
public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Constructs a biologically-inspired neural network
     * @param config Comprehensive network configuration with biological parameters
     */
    explicit Network(const NetworkConfig& config);
    
    /**
     * @brief Destructor with graceful network deconstruction
     */
    ~Network();
    
    // Prevent copying and moving due to std::random_device limitations
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) = delete;
    Network& operator=(Network&&) = delete;

    // ========================================================================
    // CORE SIMULATION INTERFACE - Brain-Like Dynamics
    // ========================================================================
    
    /**
     * @brief Updates the entire network for a single time step with biological realism
     * @param dt The simulation time step in milliseconds (typically 0.1-1.0ms)
     * @param input_currents Vector of input currents for designated input neurons
     * @param reward Global reward signal for reinforcement learning modulation
     * 
     * Performs comprehensive network update including:
     * - Neural membrane dynamics with Izhikevich model fidelity
     * - Synaptic transmission with realistic delays and dynamics
     * - Spike-timing dependent plasticity with homeostatic regulation
     * - Structural plasticity for adaptive connectivity
     * - Network-wide coordination and attention mechanisms
     */
    void update(float dt, const std::vector<float>& input_currents, float reward = 0.0f);
    
    /**
     * @brief Retrieves current network output with biological encoding
     * @return Vector of output values representing firing rates or spike patterns
     * 
     * Output encoding supports multiple biological interpretations:
     * - Population firing rates for rate-based computation
     * - Spike timing patterns for temporal coding
     * - Population vector representations for motor control
     */
    std::vector<float> get_output() const;
    
    /**
     * @brief Resets network to biological resting state while preserving structure
     * 
     * Resets membrane potentials, synaptic states, and activity traces while
     * maintaining learned synaptic weights and structural connectivity for
     * rapid task switching and transfer learning capabilities.
     */
    void reset();

    // ========================================================================
    // MODULAR NETWORK CONSTRUCTION - Dynamic Architecture Building
    // ========================================================================
    
    /**
     * @brief Adds a neuron to the network with biological parameter validation
     * @param neuron Unique pointer to biologically-configured neuron
     */
    void add_neuron(std::unique_ptr<Neuron> neuron);
    
    /**
     * @brief Adds a synapse to the network with connectivity constraints
     * @param synapse Unique pointer to biologically-configured synapse
     */
    void add_synapse(std::unique_ptr<Synapse> synapse);
    
    /**
     * @brief Creates and validates new synaptic connection
     * @param source_neuron_id Presynaptic neuron identifier
     * @param target_neuron_id Postsynaptic neuron identifier
     * @param type Synapse type ("excitatory", "inhibitory", "modulatory")
     * @param delay Transmission delay in simulation steps (1-20 typical)
     * @param weight Initial synaptic strength with biological bounds
     * @return Pointer to created synapse or nullptr if creation failed
     * 
     * Enforces biological constraints:
     * - Dale's principle for neurotransmitter consistency
     * - Realistic synaptic delays and weights
     * - Prevention of duplicate connections
     * - Distance-dependent connection probabilities
     */
    Synapse* createSynapse(size_t source_neuron_id, size_t target_neuron_id, 
                          const std::string& type, int delay, float weight);

    // ========================================================================
    // NETWORK ACCESS INTERFACE - Optimized for Modular Queries
    // ========================================================================
    
    /**
     * @brief Retrieves neuron by ID with efficient lookup
     * @param neuron_id Unique neuron identifier
     * @return Pointer to neuron or nullptr if not found
     */
    Neuron* get_neuron(size_t neuron_id) const;
    
    /**
     * @brief Retrieves synapse by ID with efficient lookup
     * @param synapse_id Unique synapse identifier  
     * @return Pointer to synapse or nullptr if not found
     */
    Synapse* get_synapse(size_t synapse_id) const;
    
    /**
     * @brief Gets all outgoing synapses from a neuron
     * @param neuron_id Source neuron identifier
     * @return Vector of synapse pointers for efficient iteration
     */
    std::vector<Synapse*> getOutgoingSynapses(size_t neuron_id) const;
    
    /**
     * @brief Gets all incoming synapses to a neuron (const-correct version)
     * @param neuron_id Target neuron identifier
     * @return Vector of synapse pointers for analysis operations
     */
    std::vector<Synapse*> getIncomingSynapses(size_t neuron_id) const;
    
    /**
     * @brief Associates network with parent neural module for hierarchical processing
     * @param module Pointer to containing neural module
     */
    void set_module(NeuralModule* module);
    
    /**
     * @brief Retrieves comprehensive network statistics
     * @return Current NetworkStats structure with biological metrics
     */
    NetworkStats get_stats() const;

    // ========================================================================
    // PERSISTENCE - Save and Load Network State
    // ========================================================================
    
    /**
     * @brief Saves the complete network state to a binary file.
     * @param file_path The path to the file where the network will be saved.
     * @return True if saving was successful, false otherwise.
     */
    bool saveToFile(const std::string& file_path) const;

    /**
     * @brief Loads the network state from a binary file.
     * @param file_path The path to the file from which to load the network.
     * @return True if loading was successful, false otherwise.
     */
    bool loadFromFile(const std::string& file_path);

    // ========================================================================
    // ADVANCED BIOLOGICAL FEATURES - Cutting-Edge Neuroscience
    // ========================================================================
    
    /**
     * @brief Enables/disables structural plasticity mechanisms
     * @param enabled Whether to allow dynamic synaptic pruning and growth
     */
    void set_structural_plasticity(bool enabled);
    
    /**
     * @brief Configures homeostatic regulation parameters
     * @param target_activity Desired network activity level (Hz)
     * @param regulation_strength Strength of homeostatic feedback (0.0-1.0)
     */
    void set_homeostatic_regulation(float target_activity, float regulation_strength);
    
    /**
     * @brief Sets attention/control signal for modular coordination
     * @param attention_weights Vector of attention weights per neural population
     */
    void set_attention_signal(const std::vector<float>& attention_weights);
    
    /**
     * @brief Configures reward learning parameters for dopaminergic modulation
     * @param learning_rate Base learning rate for reward-modulated plasticity
     * @param eligibility_decay Time constant for eligibility trace decay
     */
    void configure_reward_learning(float learning_rate, float eligibility_decay);

    // ========================================================================
    // STRUCTURAL PLASTICITY INTERFACE - Dynamic Network Architecture
    // ========================================================================
    
    /**
     * @brief Implements activity-dependent synaptic pruning
     * 
     * Biological pruning mechanisms:
     * - Use-dependent elimination of weak synapses
     * - Competitive removal based on relative strength
     * - Homeostatic adjustment of total connectivity
     * - Critical period-like developmental pruning
     */
    void prune_synapses();
    
    /**
     * @brief Implements experience-driven synaptic growth
     * 
     * Growth mechanisms:
     * - Activity-dependent formation of new connections
     * - Exploration of local connectivity space
     * - Reinforcement of successful communication pathways
     * - Homeostatic maintenance of optimal connectivity
     */
    void grow_synapses();

    // ========================================================================
    // STATE PERSISTENCE - Independent Module Saving/Loading
    // ========================================================================
    
    /**
     * @brief Saves complete network state to binary format
     * @param filename Base filename for state files
     * @return Success status of save operation
     * 
     * Saves comprehensive state including:
     * - All neuron parameters and state variables
     * - Complete synaptic weights and plasticity states
     * - Network topology and connection maps
     * - Learning state and eligibility traces
     * - Statistical measures and performance metrics
     */
    bool save_state(const std::string& filename) const;
    
    /**
     * @brief Loads complete network state from binary format
     * @param filename Base filename for state files
     * @return Success status of load operation
     * 
     * Supports rapid deployment of trained networks and
     * transfer learning between different tasks and contexts.
     */
    bool load_state(const std::string& filename);
    
    /**
     * @brief Exports network structure for visualization and analysis
     * @param filename Output file for network graph representation
     * @param format Export format ("graphml", "json", "dot")
     */
    void export_structure(const std::string& filename, const std::string& format = "graphml") const;

private:
    // ========================================================================
    // BIOLOGICAL NEURAL COMPUTATION METHODS - Advanced Neuroscience
    // ========================================================================
    
    /**
     * @brief Calculates biologically-realistic firing rate with membrane dynamics
     * @param neuron Target neuron for firing rate analysis
     * @return Current firing rate in Hz with biological constraints (0-100 Hz)
     * 
     * Incorporates:
     * - Membrane potential dynamics and threshold proximity
     * - Recent spiking history with adaptation effects
     * - Subthreshold oscillations and intrinsic excitability
     * - Biological refractory periods and rate limitations
     */
    float calculateNeuronFiringRate(const Neuron& neuron) const;
    
    /**
     * @brief Determines connection probability using biological distance rules
     * @param pre_id Presynaptic neuron identifier
     * @param post_id Postsynaptic neuron identifier
     * @return Connection probability (0.0-1.0) based on biological constraints
     * 
     * Models realistic cortical connectivity:
     * - Distance-dependent exponential decay
     * - Small-world network topology bias
     * - Layer-specific connection preferences
     * - Excitatory/inhibitory balance considerations
     */
    float calculateConnectionProbability(size_t pre_id, size_t post_id) const;
    
    /**
     * @brief Calculates total synaptic input with biological integration
     * @param neuron_id Target neuron for input summation
     * @return Total weighted synaptic input including delays and dynamics
     * 
     * Advanced synaptic integration:
     * - Realistic transmission delays and jitter
     * - Short-term synaptic dynamics (depression/facilitation)
     * - Compartmental dendritic integration
     * - Inhibitory shunting and normalization
     */
    float calculateTotalSynapticInput(size_t neuron_id) const;
    
    /**
     * @brief Advanced synaptic plasticity with multiple learning rules
     * @param synapse Target synapse for plasticity update
     * @param dt Simulation time step for integration
     * @param reward Global reward signal for reinforcement modulation
     * 
     * Implements sophisticated plasticity mechanisms:
     * - Spike-timing dependent plasticity (STDP) with biological time constants
     * - Homeostatic scaling for network stability
     * - Reward-modulated learning with dopaminergic-like signaling
     * - Metaplasticity for learning rate adaptation
     * - Heterosynaptic competition and normalization
     */
    void updateSynapticPlasticity(Synapse& synapse, float dt, float reward);
    
    /**
     * @brief Optimized synapse existence check for large-scale networks
     * @param pre_id Presynaptic neuron identifier
     * @param post_id Postsynaptic neuron identifier
     * @return True if direct synaptic connection exists
     */
    bool synapseExists(size_t pre_id, size_t post_id) const;
    
    /**
     * @brief Determines neuron activity state with multiple criteria
     * @param neuron_id Target neuron identifier
     * @return True if neuron is in active state (firing or near threshold)
     * 
     * Activity criteria:
     * - Recent spiking activity within biological time window
     * - Membrane potential proximity to spike threshold
     * - Synaptic input integration and summation
     * - Intrinsic excitability and subthreshold dynamics
     */
    bool isNeuronActive(size_t neuron_id) const;
    
    /**
     * @brief Rebuilds connection maps for structural plasticity efficiency
     * 
     * Optimizes data structures after synaptic pruning/growth for:
     * - Fast synaptic lookup during simulation
     * - Efficient connectivity queries for analysis
     * - Memory-optimal storage of sparse connectivity
     * - Cache-friendly access patterns for performance
     */
    void rebuild_connection_maps();

    // ========================================================================
    // INITIALIZATION METHODS - Biological Network Assembly
    // ========================================================================
    
    /**
     * @brief Initializes neurobiologically diverse neuron population
     * 
     * Creates realistic cortical neuron diversity:
     * - 80% excitatory regular spiking neurons (Dale's principle)
     * - 20% inhibitory fast spiking interneurons
     * - Individual parameter variability for biological realism
     * - Layer-appropriate cell type distributions
     */
    void initialize_neurons();
    
    /**
     * @brief Establishes biologically-constrained synaptic connectivity
     * 
     * Implements realistic connectivity patterns:
     * - Distance-dependent connection probabilities
     * - Small-world network topology with clustering
     * - Appropriate synaptic density and weight distributions
     * - Balanced excitation-inhibition ratios
     */
    void initialize_synapses();

    // ========================================================================
    // SIMULATION UPDATE METHODS - Core Neural Dynamics
    // ========================================================================
    
    /**
     * @brief Updates neural membrane dynamics with biological fidelity
     * @param dt Simulation time step
     * @param input_currents External input current array
     */
    void update_neurons(float dt, const std::vector<float>& input_currents);
    
    /**
     * @brief Updates synaptic transmission and plasticity
     * @param dt Simulation time step
     * @param reward Global reward signal for learning modulation
     */
    void update_synapses(float dt, float reward);
    
    /**
     * @brief Implements structural plasticity mechanisms
     * 
     * Biological structural adaptation:
     * - Activity-dependent synaptic pruning
     * - Experience-driven synaptogenesis
     * - Homeostatic connectivity regulation
     * - Critical period-like developmental dynamics
     */
    void structural_plasticity();
    
    /**
     * @brief Updates comprehensive network statistics and metrics
     * @param dt Simulation time step for rate calculations
     */
    void update_stats(float dt);

    // ========================================================================
    // ADVANCED BIOLOGICAL MECHANISMS - Cutting-Edge Features
    // ========================================================================
    
    /**
     * @brief Calculates synaptic efficacy for short-term dynamics
     * @param synapse Target synapse for efficacy calculation
     * @return Efficacy multiplier for synaptic transmission (0.1-3.0 typical)
     */
    float calculate_synaptic_efficacy(const Synapse& synapse) const;
    
    /**
     * @brief Applies homeostatic scaling for network stability
     * @param synapse Target synapse for scaling adjustment
     */
    void apply_homeostatic_scaling(Synapse& synapse);
    
    /**
     * @brief Calculates network-wide synchronization measures
     * 
     * Computes multiple synchrony metrics:
     * - Population spike coincidence
     * - Phase coherence across frequency bands
     * - Critical dynamics and avalanche distributions
     * - Information integration measures
     */
    void calculate_network_synchrony();
    
    /**
     * @brief Calculates information entropy in network activity patterns
     * 
     * Entropy measures for network analysis:
     * - Shannon entropy of population activity
     * - Temporal complexity of spike patterns
     * - Mutual information between neural populations
     * - Effective connectivity strength
     */
    void calculate_network_entropy();
    
    /**
     * @brief Coordinates activity between connected neural modules
     * 
     * Implements attention and control mechanisms:
     * - Top-down attentional modulation
     * - Bottom-up salience signaling
     * - Inter-module communication protocols
     * - Context-dependent activity routing
     */
    void coordinate_modular_activity();
    
    /**
     * @brief Applies global inhibitory modulation for stability
     * @param factor Inhibition scaling factor (0.8-1.0 typical)
     */
    void apply_global_inhibition(float factor);
    
    /**
     * @brief Applies global excitatory modulation for arousal
     * @param factor Excitation scaling factor (1.0-1.2 typical)
     */
    void apply_global_excitation(float factor);

    // ========================================================================
    // STRUCTURAL PLASTICITY METHODS - Dynamic Network Architecture
    // ========================================================================
    
    /**
     * @brief Evaluates synapse for pruning based on biological criteria
     * @param synapse Target synapse for evaluation
     * @return True if synapse should be removed
     * 
     * Pruning criteria:
     * - Low synaptic strength relative to noise floor
     * - Minimal activity over extended time periods
     * - Competition with stronger neighboring synapses
     * - Age-dependent elimination for network refinement
     */
    bool shouldPruneSynapse(const Synapse& synapse) const;

    // ========================================================================
    // MEMBER VARIABLES - Optimized Data Structures
    // ========================================================================
    
    // Core neural storage with efficient access patterns
    std::vector<std::unique_ptr<Neuron>> neurons_;
    std::vector<std::unique_ptr<Synapse>> synapses_;
    std::unordered_map<size_t, Neuron*> neuron_map_;
    
    // Connection mapping for high-performance synaptic access
    mutable std::unordered_map<size_t, std::vector<Synapse*>> incoming_synapse_map_;
    mutable std::unordered_map<size_t, std::vector<Synapse*>> outgoing_synapse_map_;
    mutable std::unordered_map<size_t, Synapse*> synapse_map_;
    
    // Random number generation for biological stochasticity
    mutable std::random_device random_device_;
    mutable std::mt19937 random_engine_;
    
    // Network configuration and performance statistics
    NetworkConfig config_;
    NetworkStats stats_;
    
    // Modular architecture support
    NeuralModule* module_;
    
    // Advanced biological state variables
    float homeostatic_target_activity_;
    float homeostatic_regulation_strength_;
    std::vector<float> attention_weights_;
    float reward_learning_rate_;
    float eligibility_decay_rate_;
    bool structural_plasticity_enabled_;
    bool homeostatic_regulation_enabled_;
    
    // Performance optimization state
    std::chrono::high_resolution_clock::time_point last_update_time_;
    float adaptive_timestep_factor_;
    size_t simulation_step_counter_;
    
    // Friend classes for advanced integration
    friend class NetworkBuilder;
    friend class NeuralModule;
    friend class GPUNeuralManager;
};

// ============================================================================
// INLINE PERFORMANCE-CRITICAL METHODS
// ============================================================================

/**
 * @brief Fast neuron lookup for simulation performance
 */
inline Neuron* Network::get_neuron(size_t neuron_id) const {
    auto it = neuron_map_.find(neuron_id);
    return (it != neuron_map_.end()) ? it->second : nullptr;
}

/**
 * @brief Fast synapse lookup for connectivity queries
 */
inline Synapse* Network::get_synapse(size_t synapse_id) const {
    auto it = synapse_map_.find(synapse_id);
    return (it != synapse_map_.end()) ? it->second : nullptr;
}

/**
 * @brief Efficient existence check for connection validation
 */
inline bool Network::synapseExists(size_t pre_id, size_t post_id) const {
    const auto& outgoing_it = outgoing_synapse_map_.find(pre_id);
    if (outgoing_it == outgoing_synapse_map_.end()) return false;
    
    const auto& synapses = outgoing_it->second;
    return std::any_of(synapses.begin(), synapses.end(),
                      [post_id](const Synapse* syn) { 
                          return syn && syn->post_neuron_id == post_id; 
                      });
}

#endif // NETWORK_H