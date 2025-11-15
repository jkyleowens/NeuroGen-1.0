// ============================================================================
// COMPREHENSIVE HEADER ARCHITECTURE FIXES
// File: include/NeuroGen/NeuralConstants.h
// ============================================================================

#ifndef NEURAL_CONSTANTS_H
#define NEURAL_CONSTANTS_H

// ============================================================================
// NEURAL RECEPTOR AND CHANNEL CONSTANTS
// ============================================================================

// Receptor type enumeration and constants
#define NUM_RECEPTOR_TYPES 8
#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3
#define RECEPTOR_GLYCINE 4
#define RECEPTOR_ACETYLCHOLINE 5
#define RECEPTOR_DOPAMINE 6
#define RECEPTOR_SEROTONIN 7

// Neural developmental stages
#define NEURAL_STAGE_PROGENITOR 0
#define NEURAL_STAGE_DIFFERENTIATION 1
#define NEURAL_STAGE_MIGRATION 2
#define NEURAL_STAGE_MATURATION 3
#define NEURAL_STAGE_ADULT 4
#define NEURAL_STAGE_SENESCENCE 5

// Compartment types
#define COMPARTMENT_SOMA 0
#define COMPARTMENT_BASAL 1
#define COMPARTMENT_APICAL 2
#define COMPARTMENT_SPINE 3
#define COMPARTMENT_AXON 4

// Maximum structure sizes
#define MAX_COMPARTMENTS 8
#define MAX_DENDRITIC_SPIKES 4
#define MAX_SYNAPSES_PER_NEURON 1000
#define MAX_NEURAL_PROGENITORS 10000

// Biophysical constants
#define RESTING_POTENTIAL -70.0f
#define SPIKE_THRESHOLD -55.0f
#define SPIKE_PEAK 30.0f
#define REVERSAL_POTENTIAL_EXCITATORY 0.0f
#define REVERSAL_POTENTIAL_INHIBITORY -70.0f

// Time constants (ms)
#define TAU_MEMBRANE 20.0f
#define TAU_CALCIUM 50.0f
#define TAU_PLASTICITY 1000.0f
#define TAU_DEVELOPMENT 86400000.0f  // 24 hours in ms

// Learning constants
#define STDP_LEARNING_RATE 0.01f
#define BCM_LEARNING_RATE 0.001f
#define HOMEOSTATIC_RATE 0.0001f
#define NEUROGENESIS_RATE 0.0001f

// Utility macros for device/host compatibility
#ifdef __CUDACC__
    #define DEVICE_HOST __device__ __host__
    #define DEVICE_ONLY __device__
    #define CUDA_MIN(a, b) fminf(a, b)
    #define CUDA_MAX(a, b) fmaxf(a, b)
#else
    #define DEVICE_HOST
    #define DEVICE_ONLY
    #define CUDA_MIN(a, b) std::min(a, b)
    #define CUDA_MAX(a, b) std::max(a, b)
#endif

#endif // NEURAL_CONSTANTS_H

// ============================================================================
// COMPLETE GPU NEURAL STRUCTURES (UPDATED)
// File: include/NeuroGen/cuda/GPUNeuralStructures.h
// ============================================================================

#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include "NeuroGen/NeuralConstants.h"

// ============================================================================
// COMPLETE GPU NEURON STATE STRUCTURE
// ============================================================================

struct GPUNeuronState {
    // === CORE MEMBRANE DYNAMICS ===
    float V;                            // Membrane potential (mV)
    float u;                            // Recovery variable (Izhikevich)
    float I_syn[MAX_COMPARTMENTS];      // Synaptic currents per compartment
    float ca_conc[MAX_COMPARTMENTS];    // Calcium concentrations
    
    // === TIMING AND ACTIVITY ===
    float last_spike_time;              // Time of last spike
    float previous_spike_time;          // Previous spike time
    float average_firing_rate;          // Running average firing rate
    float average_activity;             // Average activity level
    float activity_level;               // CRITICAL FIX: Added missing member
    
    // === PLASTICITY AND ADAPTATION ===
    float excitability;                 // Intrinsic excitability
    float synaptic_scaling_factor;      // Global synaptic scaling
    float bcm_threshold;                // BCM learning threshold
    float plasticity_threshold;         // Plasticity induction threshold
    
    // === NEUROMODULATION ===
    float dopamine_concentration;       // Local dopamine level
    float acetylcholine_level;          // Local acetylcholine level
    float serotonin_level;              // Local serotonin level
    float norepinephrine_level;         // Local norepinephrine level
    
    // === ION CHANNELS ===
    float na_m, na_h;                   // Sodium channel states
    float k_n;                          // Potassium channel state
    float ca_channel_state;             // Calcium channel state
    float channel_expression[NUM_RECEPTOR_TYPES]; // CRITICAL FIX: Added missing array
    float channel_maturation[NUM_RECEPTOR_TYPES]; // CRITICAL FIX: Added missing array
    
    // === MULTI-COMPARTMENT SUPPORT ===
    float V_compartments[MAX_COMPARTMENTS];        // Compartment voltages
    int compartment_types[MAX_COMPARTMENTS];       // Compartment types
    int num_compartments;                          // Number of active compartments
    bool dendritic_spike[MAX_DENDRITIC_SPIKES];    // Dendritic spike states
    float dendritic_spike_time[MAX_DENDRITIC_SPIKES]; // Dendritic spike timing
    
    // === NETWORK PROPERTIES ===
    int neuron_type;                    // Neuron type (excitatory/inhibitory)
    int layer_id;                       // Cortical layer
    int column_id;                      // Cortical column
    int active;                         // Activity flag
    bool is_principal_cell;             // Principal vs interneuron
    
    // === SPATIAL PROPERTIES ===
    float position_x, position_y, position_z;     // 3D coordinates
    float orientation_theta;            // Orientation
    
    // === DEVELOPMENT ===
    int developmental_stage;            // Current development stage
    float maturation_factor;            // Maturation level [0,1]
    float birth_time;                   // Time of neurogenesis
    
    // === METABOLISM ===
    float energy_level;                 // Cellular energy
    float metabolic_demand;             // Energy demand
    float glucose_uptake;               // Glucose consumption rate
};

// ============================================================================
// ENHANCED GPU SYNAPSE STRUCTURE
// ============================================================================

struct GPUSynapse {
    // === CONNECTIVITY ===
    int pre_neuron_idx;                 // Presynaptic neuron index
    int post_neuron_idx;                // Postsynaptic neuron index
    int post_compartment;               // Target compartment
    int receptor_index;                 // Receptor type
    int active;                         // Activity flag
    
    // === SYNAPTIC PROPERTIES ===
    float weight;                       // Current weight
    float max_weight, min_weight;       // Weight bounds
    float delay;                        // Synaptic delay
    float effective_weight;             // Modulated weight
    
    // === PLASTICITY ===
    float eligibility_trace;            // Eligibility trace
    float plasticity_modulation;        // Plasticity modulation
    bool is_plastic;                    // CRITICAL FIX: Added missing member
    float learning_rate;                // Synapse-specific learning rate
    float metaplasticity_factor;        // Meta-plasticity scaling
    
    // === TIMING ===
    float last_pre_spike_time;          // Last presynaptic spike
    float last_post_spike_time;         // Last postsynaptic spike
    float last_active_time;             // Last activation time
    float activity_metric;              // Activity measure
    float last_potentiation;            // Last potentiation time
    
    // === NEUROMODULATION ===
    float dopamine_sensitivity;         // Dopamine sensitivity
    float acetylcholine_sensitivity;    // ACh sensitivity
    float serotonin_sensitivity;        // Serotonin sensitivity
    float dopamine_level;               // Local dopamine
    
    // === VESICLE DYNAMICS ===
    int vesicle_count;                  // Available vesicles
    float release_probability;          // Release probability
    float facilitation_factor;          // Short-term facilitation
    float depression_factor;            // Short-term depression
    
    // === CALCIUM DYNAMICS ===
    float presynaptic_calcium;          // Pre-synaptic calcium
    float postsynaptic_calcium;         // Post-synaptic calcium
    
    // === HOMEOSTASIS ===
    float homeostatic_scaling;          // Homeostatic scaling
    float target_activity;              // Target activity level
    
    // === BIOPHYSICS ===
    float conductance;                  // Synaptic conductance
    float reversal_potential;           // Reversal potential
    float time_constant_rise;           // Rise time constant
    float time_constant_decay;          // Decay time constant
    
    // === DEVELOPMENT ===
    int developmental_stage;            // Development stage
    float structural_stability;         // Resistance to pruning
    float growth_factor;                // Growth tendency
};

// ============================================================================
// FORWARD DECLARATIONS FOR COMPLEX TYPES
// ============================================================================

// Value function approximation structure
struct ValueFunction {
    float state_features[64];           // State feature representation
    float value_weights[64];            // Value function weights
    float state_value;                  // Current state value estimate
    float td_error;                     // Temporal difference error
    float learning_rate;                // Value function learning rate
    float eligibility_trace;            // Eligibility trace for TD learning
    int feature_dimensions;             // Number of active features
    bool is_active;                     // Whether this function is active
};

// Actor-critic learning structure
struct ActorCriticState {
    float policy_parameters[32];        // Policy parameters
    float action_probabilities[32];     // Action probabilities
    float action_preferences[32];       // Action preferences
    float action_eligibility[32];       // Action eligibility traces
    float state_value;                  // State value estimate
    float baseline_estimate;            // Baseline for advantage
    float advantage_estimate;           // Advantage estimate
    float exploration_bonus;            // Exploration bonus
    float uncertainty_estimate;         // Epistemic uncertainty
    int num_actions;                    // Number of possible actions
    bool is_learning;                   // Learning flag
};

// Curiosity-driven exploration system
struct CuriosityState {
    float novelty_detector[32];         // Novelty detection features
    float surprise_level;               // Current surprise level
    float familiarity_level;            // Familiarity with current state
    float information_gain;             // Expected information gain
    float competence_progress;          // Learning progress measure
    float mastery_level;                // Current mastery level
    float random_exploration;           // Random exploration drive
    float directed_exploration;         // Directed exploration drive
    float goal_exploration;             // Goal-directed exploration
    bool is_exploring;                  // Exploration flag
};

// Neural progenitor cell structure
struct NeuralProgenitor {
    // === PROGENITOR IDENTITY ===
    int progenitor_id;                  // Unique identifier
    int progenitor_type;                // Type of progenitor
    int developmental_stage;            // Current stage
    
    // === SPATIAL PROPERTIES ===
    float position_x, position_y, position_z;     // 3D coordinates
    float migration_vector_x, migration_vector_y, migration_vector_z; // Migration direction
    
    // === TEMPORAL PROPERTIES ===
    float birth_time;                   // Time of creation
    float differentiation_time;         // Time of differentiation
    float last_division_time;           // Last division time
    
    // === PROLIFERATION ===
    int division_count;                 // Number of divisions
    int max_divisions;                  // Maximum allowed divisions
    float division_probability;         // Probability of division
    bool can_divide;                    // Division capability
    
    // === DIFFERENTIATION ===
    float differentiation_probability;  // Probability of differentiation
    float excitatory_bias;              // Bias toward excitatory fate
    float inhibitory_bias;              // Bias toward inhibitory fate
    float interneuron_probability;      // Probability of interneuron fate
    
    // === ENVIRONMENTAL SENSING ===
    float local_activity_level;         // Local network activity
    float local_neuron_density;         // Local neuron density
    float growth_factor_concentration;  // Growth factor levels
    float competition_pressure;         // Competition from other cells
    
    // === MOLECULAR STATE ===
    float transcription_factors[8];     // Key transcription factors
    float growth_signals[4];            // Growth signaling molecules
    float apoptosis_signals[4];         // Cell death signals
    
    // === FATE SPECIFICATION ===
    int target_layer;                   // Target cortical layer
    int target_column;                  // Target cortical column
    int target_neuron_type;             // Target neuron type
    bool fate_committed;                // Whether fate is determined
    
    // === ACTIVITY STATE ===
    bool is_active;                     // Whether progenitor is active
    bool is_migrating;                  // Currently migrating
    bool is_differentiating;            // Currently differentiating
    bool marked_for_deletion;           // Scheduled for removal
};

// Additional forward declarations
struct DevelopmentalTrajectory;
struct SynapticProgenitor;
struct SynapticCompetition;
struct PruningAssessment;
struct CompetitiveElimination;
struct NeuralHomeostasis;
struct SynapticHomeostasis;
struct NetworkHomeostasis;
struct STDPRuleConfig;
struct NeurogenesisController;
struct SynaptogenesisController;
struct PruningController;
struct CoordinationController;
struct PlasticityState;
struct DopamineNeuron;

#endif // GPU_NEURAL_STRUCTURES_H

// ============================================================================
// FIXED ENHANCED STDP FRAMEWORK HEADER
// File: include/NeuroGen/EnhancedSTDPFramework.h  
// ============================================================================

#ifndef ENHANCED_STDP_FRAMEWORK_H
#define ENHANCED_STDP_FRAMEWORK_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include "NeuroGen/NeuralConstants.h"

// Forward declarations (no CUDA dependencies)
struct GPUSynapse;
struct GPUNeuronState;
struct GPUPlasticityState;
struct GPUNeuromodulatorState;

/**
 * @brief Enhanced STDP Framework with Comprehensive Biological Realism
 * 
 * This framework implements multiple forms of synaptic plasticity that mirror
 * the sophistication found in biological neural circuits, including:
 * - Classical STDP with precise timing dependence
 * - BCM learning rule with sliding threshold
 * - Homeostatic synaptic scaling
 * - Neuromodulatory influences (dopamine, acetylcholine, serotonin)
 * - Meta-plasticity mechanisms
 * - Multi-compartment dendritic integration
 */
class EnhancedSTDPFramework {
private:
    // GPU memory management (using void* to avoid CUDA types)
    void* d_synapses_;
    void* d_neurons_;
    void* d_plasticity_states_;
    void* d_neuromodulator_states_;
    void* d_stdp_config_;
    
    // Network dimensions
    int num_synapses_;
    int num_neurons_;
    bool cuda_initialized_;
    
    // Learning parameters
    float stdp_learning_rate_;
    float bcm_learning_rate_;
    float homeostatic_rate_;
    float neuromodulation_strength_;
    float metaplasticity_rate_;
    
    // Performance tracking
    mutable std::mutex framework_mutex_;
    mutable float total_weight_change_;
    mutable float plasticity_events_;
    mutable float last_update_time_;
    mutable float average_eligibility_trace_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    EnhancedSTDPFramework();
    ~EnhancedSTDPFramework();
    
    /**
     * @brief Initialize the STDP framework with network specifications
     * @param num_neurons Total number of neurons in the network
     * @param num_synapses Total number of synapses in the network
     * @return Success status of initialization
     */
    bool initialize(int num_neurons, int num_synapses);
    
    /**
     * @brief Configure comprehensive learning parameters
     * @param stdp_rate STDP learning rate for spike-timing dependent plasticity
     * @param bcm_rate BCM learning rate for threshold adaptation
     * @param homeostatic_rate Rate of homeostatic regulation
     * @param neuromod_strength Strength of neuromodulatory influences
     */
    void configure_learning_parameters(float stdp_rate, float bcm_rate, 
                                     float homeostatic_rate, float neuromod_strength);
    
    // ========================================================================
    // MAIN PLASTICITY MECHANISMS - C++ WRAPPER INTERFACE
    // ========================================================================
    
    /**
     * @brief Execute enhanced STDP with multi-factor modulation
     * @param current_time Current simulation time (milliseconds)
     * @param dt Integration time step (milliseconds)
     */
    void update_enhanced_stdp(float current_time, float dt);
    
    /**
     * @brief Apply BCM learning rule with adaptive threshold
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_bcm_learning(float current_time, float dt);
    
    /**
     * @brief Execute homeostatic regulation mechanisms
     * @param target_activity Target network activity level (Hz)
     * @param dt Time step
     */
    void update_homeostatic_regulation(float target_activity, float dt);
    
    /**
     * @brief Apply neuromodulatory influences on plasticity
     * @param dopamine_level Global dopamine concentration
     * @param acetylcholine_level Global acetylcholine level
     * @param dt Time step
     */
    void update_neuromodulation(float dopamine_level, float acetylcholine_level, float dt);
    
    /**
     * @brief Update meta-plasticity mechanisms
     * @param experience_level Current experience level
     * @param dt Time step
     */
    void update_metaplasticity(float experience_level, float dt);
    
    /**
     * @brief Execute comprehensive plasticity update (all mechanisms)
     * @param current_time Current simulation time
     * @param dt Time step
     * @param dopamine_level Dopamine concentration
     * @param target_activity Target activity level
     */
    void update_all_plasticity_mechanisms(float current_time, float dt, 
                                        float dopamine_level, float target_activity);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get total synaptic weight change magnitude
     * @return Cumulative weight change since last reset
     */
    float get_total_weight_change() const;
    
    /**
     * @brief Get number of plasticity events
     * @return Count of significant plasticity events
     */
    float get_plasticity_events() const;
    
    /**
     * @brief Get average synaptic weight across network
     * @return Mean synaptic weight
     */
    float get_average_synaptic_weight() const;
    
    /**
     * @brief Get average eligibility trace magnitude
     * @return Mean eligibility trace value
     */
    float get_average_eligibility_trace() const;
    
    /**
     * @brief Get comprehensive plasticity statistics
     * @param stats Output vector containing detailed statistics
     */
    void get_plasticity_statistics(std::vector<float>& stats) const;
    
    /**
     * @brief Generate detailed plasticity report
     * @param filename Output filename for detailed report
     */
    void generate_plasticity_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save complete plasticity state to file
     * @param filename Base filename for state persistence
     * @return Success status
     */
    bool save_plasticity_state(const std::string& filename) const;
    
    /**
     * @brief Load complete plasticity state from file
     * @param filename Base filename for state loading
     * @return Success status
     */
    bool load_plasticity_state(const std::string& filename);
    
    /**
     * @brief Reset all plasticity mechanisms to baseline
     */
    void reset_plasticity_state();
    
    /**
     * @brief Set plasticity mechanism enable/disable flags
     * @param enable_stdp Enable STDP mechanisms
     * @param enable_bcm Enable BCM learning
     * @param enable_homeostatic Enable homeostatic regulation
     * @param enable_neuromodulation Enable neuromodulatory influences
     */
    void configure_plasticity_mechanisms(bool enable_stdp, bool enable_bcm, 
                                       bool enable_homeostatic, bool enable_neuromodulation);

private:
    // ========================================================================
    // INTERNAL CUDA WRAPPER FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources and memory allocation
     * @return Success status of CUDA initialization
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Release all CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Launch enhanced STDP kernel (internal wrapper)
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void launch_enhanced_stdp_kernel(float current_time, float dt);
    
    /**
     * @brief Launch BCM learning kernel (internal wrapper)
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void launch_bcm_learning_kernel(float current_time, float dt);
    
    /**
     * @brief Launch homeostatic regulation kernel (internal wrapper)
     * @param target_activity Target activity level
     * @param dt Time step
     */
    void launch_homeostatic_kernel(float target_activity, float dt);
    
    /**
     * @brief Launch neuromodulation kernel (internal wrapper)
     * @param dopamine_level Dopamine concentration
     * @param acetylcholine_level Acetylcholine level
     * @param dt Time step
     */
    void launch_neuromodulation_kernel(float dopamine_level, float acetylcholine_level, float dt);
    
    /**
     * @brief Update performance metrics from GPU
     */
    void update_performance_metrics();
    
    /**
     * @brief Validate CUDA operation success
     * @param operation_name Operation name for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
    
    /**
     * @brief Copy statistics from GPU to CPU
     */
    void copy_statistics_from_gpu();
    
    /**
     * @brief Configure optimal GPU execution parameters
     */
    void configure_gpu_execution_parameters();
};

#endif // ENHANCED_STDP_FRAMEWORK_H

// ============================================================================
// FIXED ADVANCED REINFORCEMENT LEARNING HEADER
// File: include/NeuroGen/AdvancedReinforcementLearning.h
// ============================================================================

#ifndef ADVANCED_REINFORCEMENT_LEARNING_H
#define ADVANCED_REINFORCEMENT_LEARNING_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include "NeuroGen/NeuralConstants.h"

// Forward declarations (avoiding CUDA dependencies)
struct GPUNeuronState;
struct ActorCriticState;
struct CuriosityState;
struct ValueFunction;
struct DopamineNeuron;

/**
 * @brief Advanced Reinforcement Learning with Biologically-Inspired Mechanisms
 * 
 * This system implements sophisticated reinforcement learning algorithms that
 * mirror the computational principles found in biological reward systems:
 * - Dopaminergic prediction error computation
 * - Actor-critic architecture with eligibility traces
 * - Curiosity-driven intrinsic motivation
 * - Multi-timescale learning (seconds to hours)
 * - Experience replay and consolidation
 * - Meta-learning and adaptation
 */
class AdvancedReinforcementLearning {
private:
    // GPU memory management (void* to avoid CUDA dependencies)
    void* d_da_neurons_;
    void* d_actor_critic_states_;
    void* d_curiosity_states_;
    void* d_value_functions_;
    void* d_reward_signals_;
    void* d_experience_buffer_;
    
    // Network parameters
    int num_da_neurons_;
    int num_network_neurons_;
    int num_value_functions_;
    bool cuda_initialized_;
    
    // Learning parameters
    float actor_learning_rate_;
    float critic_learning_rate_;
    float curiosity_strength_;
    float exploration_noise_;
    float experience_replay_rate_;
    
    // Performance tracking
    mutable std::mutex rl_mutex_;
    float total_reward_;
    float prediction_accuracy_;
    int learning_updates_;
    float exploration_efficiency_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    AdvancedReinforcementLearning();
    ~AdvancedReinforcementLearning();
    
    /**
     * @brief Initialize reinforcement learning system
     * @param num_da_neurons Number of dopamine neurons
     * @param num_network_neurons Total network neurons for modulation
     * @param num_value_functions Number of parallel value functions
     * @return Success status
     */
    bool initialize(int num_da_neurons, int num_network_neurons, int num_value_functions = 1);
    
    /**
     * @brief Configure comprehensive learning parameters
     * @param actor_lr Actor network learning rate
     * @param critic_lr Critic network learning rate
     * @param curiosity_strength Intrinsic motivation strength
     * @param exploration_noise Exploration noise magnitude
     */
    void configure_learning_parameters(float actor_lr, float critic_lr, 
                                     float curiosity_strength, float exploration_noise);
    
    // ========================================================================
    // MAIN LEARNING MECHANISMS - C++ WRAPPER INTERFACE
    // ========================================================================
    
    /**
     * @brief Update dopaminergic system with reward processing
     * @param reward_signal Current environmental reward
     * @param predicted_reward Value function prediction
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_dopaminergic_system(float reward_signal, float predicted_reward, 
                                   float current_time, float dt);
    
    /**
     * @brief Update actor-critic learning with experience
     * @param state_features Current state representation
     * @param action_taken Action that was executed
     * @param reward_received Reward from environment
     * @param next_state_features Subsequent state representation
     * @param dt Time step
     */
    void update_actor_critic(const std::vector<float>& state_features, int action_taken,
                           float reward_received, const std::vector<float>& next_state_features, 
                           float dt);
    
    /**
     * @brief Update curiosity-driven exploration system
     * @param environmental_features Current environmental observations
     * @param prediction_error Prediction error magnitude
     * @param dt Time step
     */
    void update_curiosity_system(const std::vector<float>& environmental_features,
                                float prediction_error, float dt);
    
    /**
     * @brief Execute experience replay for consolidation
     * @param num_replay_samples Number of experiences to replay
     * @param dt Time step
     */
    void execute_experience_replay(int num_replay_samples, float dt);
    
    /**
     * @brief Update value function approximation
     * @param state_features Current state features
     * @param target_value Target value for update
     * @param dt Time step
     */
    void update_value_function(const std::vector<float>& state_features, 
                              float target_value, float dt);
    
    // ========================================================================
    // DECISION MAKING INTERFACE
    // ========================================================================
    
    /**
     * @brief Get action probabilities from actor network
     * @param state_features Current state representation
     * @return Probability distribution over actions
     */
    std::vector<float> get_action_probabilities(const std::vector<float>& state_features) const;
    
    /**
     * @brief Get state value estimate from critic network
     * @param state_features Current state representation
     * @return Estimated state value
     */
    float get_state_value_estimate(const std::vector<float>& state_features) const;
    
    /**
     * @brief Get intrinsic motivation signal
     * @param state_features Current state
     * @return Curiosity-driven reward bonus
     */
    float get_intrinsic_motivation(const std::vector<float>& state_features) const;
    
    /**
     * @brief Sample action from policy
     * @param state_features Current state
     * @param exploration_factor Exploration intensity [0,1]
     * @return Selected action index
     */
    int sample_action(const std::vector<float>& state_features, float exploration_factor = 1.0f) const;
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get current dopamine levels across DA neurons
     * @return Vector of dopamine concentrations
     */
    std::vector<float> get_dopamine_levels() const;
    
    /**
     * @brief Get comprehensive learning performance metrics
     * @param metrics Output vector containing performance data
     */
    void get_performance_metrics(std::vector<float>& metrics) const;
    
    /**
     * @brief Get total accumulated reward
     * @return Cumulative reward received
     */
    float get_total_reward() const;
    
    /**
     * @brief Get prediction accuracy of value functions
     * @return Current prediction accuracy [0,1]
     */
    float get_prediction_accuracy() const;
    
    /**
     * @brief Get exploration efficiency metric
     * @return Exploration efficiency measure
     */
    float get_exploration_efficiency() const;
    
    /**
     * @brief Generate detailed learning report
     * @param filename Output filename for detailed analysis
     */
    void generate_learning_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save complete RL state to file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool save_rl_state(const std::string& filename) const;
    
    /**
     * @brief Load complete RL state from file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool load_rl_state(const std::string& filename);
    
    /**
     * @brief Reset all learning state to baseline
     */
    void reset_learning_state();

private:
    // ========================================================================
    // INTERNAL CUDA WRAPPER FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources for RL
     * @return Success status
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Cleanup all CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Launch dopamine update kernel (internal wrapper)
     * @param reward_signal Reward signal
     * @param predicted_reward Predicted reward
     * @param current_time Current time
     * @param dt Time step
     */
    void launch_dopamine_update_kernel(float reward_signal, float predicted_reward,
                                     float current_time, float dt);
    
    /**
     * @brief Launch actor-critic update kernel (internal wrapper)
     * @param state_features State features
     * @param action_taken Action taken
     * @param reward_received Reward received
     * @param dt Time step
     */
    void launch_actor_critic_kernel(const std::vector<float>& state_features, 
                                   int action_taken, float reward_received, float dt);
    
    /**
     * @brief Launch curiosity update kernel (internal wrapper)
     * @param environmental_features Environmental features
     * @param prediction_error Prediction error
     * @param dt Time step
     */
    void launch_curiosity_kernel(const std::vector<float>& environmental_features,
                                float prediction_error, float dt);
    
    /**
     * @brief Copy results from GPU to CPU
     */
    void copy_results_from_gpu();
    
    /**
     * @brief Validate CUDA operations
     * @param operation_name Operation name for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
    
    /**
     * @brief Store experience in replay buffer
     * @param state Current state
     * @param action Action taken
     * @param reward Reward received
     * @param next_state Next state
     */
    void store_experience(const std::vector<float>& state, int action, 
                         float reward, const std::vector<float>& next_state);
};

#endif // ADVANCED_REINFORCEMENT_LEARNING_H

// ============================================================================
// FIXED DYNAMIC NEUROGENESIS FRAMEWORK HEADER
// File: include/NeuroGen/DynamicNeurogenesisFramework.h
// ============================================================================

#ifndef DYNAMIC_NEUROGENESIS_FRAMEWORK_H
#define DYNAMIC_NEUROGENESIS_FRAMEWORK_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include "NeuroGen/NeuralConstants.h"

// Forward declarations (no CUDA dependencies)
struct GPUNeuronState;
struct NeuralProgenitor;
struct DevelopmentalTrajectory;
struct ValueFunction;

/**
 * @brief Dynamic Neurogenesis Framework - C++ Only Interface
 * 
 * Implements biologically-realistic neurogenesis mechanisms including:
 * - Neural progenitor cell dynamics
 * - Activity-dependent neurogenesis
 * - Developmental trajectory control
 * - Spatial organization and migration
 * - Experience-dependent plasticity
 */
class DynamicNeurogenesisFramework {
private:
    // GPU memory management (void* to avoid CUDA dependencies)
    void* d_neural_progenitors_;
    void* d_neurons_;
    void* d_developmental_trajectories_;
    void* d_value_functions_;
    void* d_neurogenesis_controller_;
    
    // Network parameters
    int max_progenitors_;
    int current_progenitors_;
    int max_neurons_;
    int current_neurons_;
    bool cuda_initialized_;
    
    // Neurogenesis parameters
    float neurogenesis_rate_;
    float activity_threshold_;
    float spatial_competition_radius_;
    float experience_dependence_;
    
    // Performance tracking
    mutable std::mutex neurogenesis_mutex_;
    int neurons_generated_;
    int progenitors_created_;
    float average_maturation_time_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    DynamicNeurogenesisFramework();
    ~DynamicNeurogenesisFramework();
    
    /**
     * @brief Initialize neurogenesis framework
     * @param max_progenitors Maximum number of progenitor cells
     * @param max_neurons Maximum number of mature neurons
     * @param initial_neurogenesis_rate Initial rate of neurogenesis
     * @return Success status
     */
    bool initialize(int max_progenitors, int max_neurons, float initial_neurogenesis_rate);
    
    /**
     * @brief Configure neurogenesis parameters
     * @param neurogenesis_rate Rate of new neuron generation
     * @param activity_threshold Activity threshold for neurogenesis
     * @param spatial_radius Spatial competition radius
     * @param experience_dep Experience dependence factor
     */
    void configure_neurogenesis_parameters(float neurogenesis_rate, float activity_threshold,
                                         float spatial_radius, float experience_dep);
    
    // ========================================================================
    // MAIN NEUROGENESIS MECHANISMS - C++ WRAPPER INTERFACE
    // ========================================================================
    
    /**
     * @brief Update neurogenesis control mechanisms
     * @param current_time Current simulation time
     * @param dt Time step
     * @param global_activity_level Overall network activity
     */
    void update_neurogenesis_control(float current_time, float dt, float global_activity_level);
    
    /**
     * @brief Execute progenitor cell spawning
     * @param current_time Current simulation time
     * @param dt Time step
     * @param environmental_factors Environmental influences
     */
    void update_progenitor_spawning(float current_time, float dt, 
                                   const std::vector<float>& environmental_factors);
    
    /**
     * @brief Update progenitor cell development
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_progenitor_development(float current_time, float dt);
    
    /**
     * @brief Execute neuron differentiation from progenitors
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_neuron_differentiation(float current_time, float dt);
    
    /**
     * @brief Update developmental trajectories
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_developmental_trajectories(float current_time, float dt);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get current number of active progenitors
     * @return Number of progenitor cells
     */
    int get_current_progenitor_count() const;
    
    /**
     * @brief Get current number of mature neurons
     * @return Number of differentiated neurons
     */
    int get_current_neuron_count() const;
    
    /**
     * @brief Get neurogenesis rate statistics
     * @param stats Output vector for neurogenesis statistics
     */
    void get_neurogenesis_statistics(std::vector<float>& stats) const;
    
    /**
     * @brief Get developmental stage distribution
     * @return Vector of counts per developmental stage
     */
    std::vector<int> get_developmental_stage_distribution() const;
    
    /**
     * @brief Generate neurogenesis report
     * @param filename Output filename for detailed report
     */
    void generate_neurogenesis_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save neurogenesis state to file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool save_neurogenesis_state(const std::string& filename) const;
    
    /**
     * @brief Load neurogenesis state from file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool load_neurogenesis_state(const std::string& filename);
    
    /**
     * @brief Reset neurogenesis to baseline state
     */
    void reset_neurogenesis_state();

private:
    // ========================================================================
    // INTERNAL CUDA WRAPPER FUNCTIONS (NO DEVICE CODE IN HEADER)
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources for neurogenesis
     * @return Success status
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Cleanup CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Launch neurogenesis control kernel (wrapper)
     * @param current_time Current time
     * @param dt Time step
     * @param global_activity Global activity level
     */
    void launch_neurogenesis_control_kernel(float current_time, float dt, float global_activity);
    
    /**
     * @brief Launch progenitor spawning kernel (wrapper)
     * @param current_time Current time
     * @param dt Time step
     * @param environmental_factors Environmental influences
     */
    void launch_progenitor_spawning_kernel(float current_time, float dt,
                                          const std::vector<float>& environmental_factors);
    
    /**
     * @brief Copy statistics from GPU to CPU
     */
    void copy_statistics_from_gpu();
    
    /**
     * @brief Validate CUDA operations
     * @param operation_name Operation name for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
};

#endif // DYNAMIC_NEUROGENESIS_FRAMEWORK_H

// ============================================================================
// FIXED PHASE 3 INTEGRATION FRAMEWORK HEADER
// File: include/NeuroGen/Phase3IntegrationFramework.h
// ============================================================================

#ifndef PHASE3_INTEGRATION_FRAMEWORK_H
#define PHASE3_INTEGRATION_FRAMEWORK_H

#include "NeuroGen/EnhancedSTDPFramework.h"
#include "NeuroGen/AdvancedReinforcementLearning.h"
#include "NeuroGen/DynamicNeurogenesisFramework.h"
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <chrono>

/**
 * @brief Phase 3 Integration Framework - Complete Neural System Integration
 * 
 * This framework integrates all advanced neural mechanisms into a unified
 * system that exhibits emergent intelligence through the interaction of:
 * - Multi-factor synaptic plasticity
 * - Dopaminergic reinforcement learning
 * - Activity-dependent neurogenesis
 * - Homeostatic regulation
 * - Meta-learning and adaptation
 */
class Phase3IntegrationFramework {
private:
    // Component frameworks
    std::unique_ptr<EnhancedSTDPFramework> stdp_framework_;
    std::unique_ptr<AdvancedReinforcementLearning> rl_framework_;
    std::unique_ptr<DynamicNeurogenesisFramework> neurogenesis_framework_;
    
    // Network parameters
    int num_neurons_;
    int num_synapses_;
    int num_modules_;
    
    // Integration state
    bool is_initialized_;
    bool cuda_available_;
    bool real_time_mode_;
    
    // Performance metrics
    float integration_efficiency_;
    float overall_learning_rate_;
    std::chrono::steady_clock::time_point last_update_;
    
    // Thread safety
    mutable std::mutex framework_mutex_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    Phase3IntegrationFramework();
    ~Phase3IntegrationFramework();
    
    /**
     * @brief Initialize the complete Phase 3 framework
     * @param num_neurons Number of neurons
     * @param num_synapses Number of synapses
     * @param num_modules Number of neural modules
     * @return Success status
     */
    bool initialize(int num_neurons, int num_synapses, int num_modules);
    
    /**
     * @brief Configure comprehensive learning parameters
     * @param stdp_rate STDP learning rate
     * @param rl_rate Reinforcement learning rate
     * @param neurogenesis_rate Neurogenesis rate
     * @param integration_strength Framework integration strength
     */
    void configure_learning_parameters(float stdp_rate, float rl_rate, 
                                     float neurogenesis_rate, float integration_strength);
    
    // ========================================================================
    // MAIN INTEGRATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Execute complete learning update across all frameworks
     * @param current_time Current simulation time
     * @param dt Time step
     * @param reward_signal Environmental reward signal
     * @param environmental_features Current environmental state
     */
    void update_integrated_learning(float current_time, float dt, float reward_signal,
                                   const std::vector<float>& environmental_features);
    
    /**
     * @brief Update plasticity mechanisms across all components
     * @param current_time Current time
     * @param dt Time step
     * @param dopamine_level Dopamine concentration
     */
    void update_plasticity_mechanisms(float current_time, float dt, float dopamine_level);
    
    /**
     * @brief Update reinforcement learning components
     * @param state_features Current state
     * @param action_taken Action executed
     * @param reward_received Reward from environment
     * @param dt Time step
     */
    void update_reinforcement_learning(const std::vector<float>& state_features, 
                                     int action_taken, float reward_received, float dt);
    
    /**
     * @brief Update neurogenesis and structural plasticity
     * @param current_time Current time
     * @param dt Time step
     * @param global_activity Global network activity
     */
    void update_structural_plasticity(float current_time, float dt, float global_activity);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get comprehensive learning statistics
     * @param stats Output vector for all statistics
     */
    void get_comprehensive_statistics(std::vector<float>& stats) const;
    
    /**
     * @brief Get integration efficiency metric
     * @return Integration efficiency [0,1]
     */
    float get_integration_efficiency() const;
    
    /**
     * @brief Get overall learning progress
     * @return Learning progress metric
     */
    float get_overall_learning_progress() const;
    
    /**
     * @brief Generate comprehensive system report
     * @param filename Output filename for detailed analysis
     */
    void generate_system_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save complete framework state
     * @param filename Base filename for state files
     * @return Success status
     */
    bool save_complete_state(const std::string& filename) const;
    
    /**
     * @brief Load complete framework state
     * @param filename Base filename for state files
     * @return Success status
     */
    bool load_complete_state(const std::string& filename);
    
    /**
     * @brief Reset all frameworks to baseline state
     */
    void reset_all_frameworks();

private:
    // ========================================================================
    // INTERNAL INTEGRATION METHODS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources for all frameworks
     * @return Success status
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Cleanup all CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Synchronize learning between frameworks
     */
    void synchronize_framework_learning();
    
    /**
     * @brief Update integration metrics
     */
    void update_integration_metrics();
    
    /**
     * @brief Validate framework states
     * @return Validation success
     */
    bool validate_framework_states() const;
    
    /**
     * @brief Coordinate framework interactions
     * @param current_time Current time
     * @param dt Time step
     */
    void coordinate_framework_interactions(float current_time, float dt);
    
    /**
     * @brief Handle framework errors and recovery
     */
    void handle_framework_errors();
};

#endif // PHASE3_INTEGRATION_FRAMEWORK_H