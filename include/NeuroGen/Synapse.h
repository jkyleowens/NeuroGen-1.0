#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Forward declaration consistency - FIX: Use class to match Neuron.h
class Neuron;

/**
 * @brief Synapse structure for neural connections - Compatible with Network.h
 * 
 * This structure maintains compatibility with the existing Network implementation
 * while providing biological properties including plasticity, delays, and
 * neurotransmitter dynamics.
 */
struct Synapse {
    // ========================================================================
    // CORE IDENTIFICATION AND TOPOLOGY (Required by Network.h)
    // ========================================================================
    size_t id;                           // Unique synapse identifier
    size_t pre_neuron_id;               // Presynaptic neuron ID
    size_t post_neuron_id;              // Postsynaptic neuron ID - REQUIRED by Network.h
    std::string post_compartment;        // Target compartment name
    size_t receptor_index;               // Receptor type index
    
    // ========================================================================
    // SYNAPTIC PROPERTIES
    // ========================================================================
    double weight;                       // Current synaptic weight/strength
    double base_weight;                  // Original/baseline weight
    double axonal_delay;                 // Transmission delay in milliseconds
    
    // ========================================================================
    // PLASTICITY PARAMETERS
    // ========================================================================
    double last_pre_spike;               // Time of last presynaptic spike
    double last_post_spike;              // Time of last postsynaptic spike
    double eligibility_trace;            // Eligibility trace for STDP
    double activity_metric;              // Running average of usage
    
    // ========================================================================
    // STRUCTURAL PLASTICITY
    // ========================================================================
    double formation_time;               // When synapse was created
    double last_potentiation;            // Last time weight was increased
    double strength_history[10];         // Sliding window for pruning decisions
    size_t history_index;                // Current position in history buffer
    
    // ========================================================================
    // NEUROTRANSMITTER DYNAMICS
    // ========================================================================
    enum class NeurotransmitterType {
        GLUTAMATE,    // Excitatory
        GABA,         // Inhibitory
        DOPAMINE,     // Modulatory
        SEROTONIN,    // Modulatory
        ACETYLCHOLINE // Modulatory
    };
    
    NeurotransmitterType neurotransmitter_type;
    float vesicle_count;                 // Available neurotransmitter vesicles
    float release_probability;           // Probability of vesicle release
    float reuptake_rate;                 // Rate of neurotransmitter reuptake
    
    // ========================================================================
    // ACTIVITY TRACKING
    // ========================================================================
    float last_spike_time;               // Time of last presynaptic spike
    bool is_active;                      // Whether synapse is currently active
    
    // ========================================================================
    // STRUCTURAL PROPERTIES
    // ========================================================================
    float spine_density;                 // Dendritic spine density
    float axon_diameter;                 // Axon diameter (affects delay)
    bool is_myelinated;                  // Whether axon is myelinated
    
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================
    
    /**
     * @brief Default constructor for container compatibility
     */
    Synapse() : 
        id(0),
        pre_neuron_id(0),
        post_neuron_id(0),
        post_compartment("soma"),
        receptor_index(0),
        weight(0.1),
        base_weight(0.1),
        axonal_delay(1.0),
        last_pre_spike(-1000.0),
        last_post_spike(-1000.0),
        eligibility_trace(0.0),
        activity_metric(0.0),
        formation_time(0.0),
        last_potentiation(0.0),
        history_index(0),
        neurotransmitter_type(NeurotransmitterType::GLUTAMATE),
        vesicle_count(100.0f),
        release_probability(0.3f),
        reuptake_rate(0.1f),
        last_spike_time(-1.0f),
        is_active(false),
        spine_density(1.0f),
        axon_diameter(1.0f),
        is_myelinated(false) {
        std::fill(std::begin(strength_history), std::end(strength_history), 0.1);
    }
    
    /**
     * @brief Full constructor compatible with Network.h requirements
     */
    Synapse(size_t synapse_id, size_t pre_id, size_t post_id, 
           const std::string& compartment, size_t receptor_idx, 
           double w = 0.1, double delay = 1.0) :
        id(synapse_id),
        pre_neuron_id(pre_id),
        post_neuron_id(post_id),
        post_compartment(compartment),
        receptor_index(receptor_idx),
        weight(w),
        base_weight(w),
        axonal_delay(delay),
        last_pre_spike(-1000.0),
        last_post_spike(-1000.0),
        eligibility_trace(0.0),
        activity_metric(0.0),
        formation_time(0.0),
        last_potentiation(0.0),
        history_index(0),
        neurotransmitter_type(NeurotransmitterType::GLUTAMATE),
        vesicle_count(100.0f),
        release_probability(0.3f),
        reuptake_rate(0.1f),
        last_spike_time(-1.0f),
        is_active(false),
        spine_density(1.0f),
        axon_diameter(1.0f),
        is_myelinated(false) {
        std::fill(std::begin(strength_history), std::end(strength_history), w);
    }
    
    // ========================================================================
    // MEMBER FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Update synaptic weight based on pre/post activity
     */
    void update_weight(float pre_activity, float post_activity, float dt);
    
    /**
     * @brief Transmit spike with biological dynamics
     */
    void transmit_spike(float current_time);
    
    /**
     * @brief Compute transmission delay based on biological parameters
     */
    float compute_transmission_delay() const;
    
    /**
     * @brief Update neurotransmitter dynamics
     */
    void update_neurotransmitter_dynamics(float dt);
    
    /**
     * @brief Reset activity counters
     */
    void reset_activity();
    
    /**
     * @brief Check if synapse should be pruned
     */
    bool should_be_pruned(float pruning_threshold = 0.01f) const;
    
    // ========================================================================
    // SERIALIZATION
    // ========================================================================
    
    /**
     * @brief Serialize synapse state to string
     */
    std::string serialize() const;
    
    /**
     * @brief Deserialize synapse state from string
     */
    bool deserialize(const std::string& data);
};

// ============================================================================
// UTILITY FUNCTIONS FOR SYNAPSE MANAGEMENT
// ============================================================================

namespace SynapseUtils {
    /**
     * @brief Create synapse with biological parameters
     */
    std::shared_ptr<Synapse> createBiologicalSynapse(
        size_t synapse_id,
        size_t pre_neuron_id,
        size_t post_neuron_id,
        Synapse::NeurotransmitterType type = Synapse::NeurotransmitterType::GLUTAMATE);
    
    /**
     * @brief Compute optimal synaptic delay based on distance
     */
    float computeOptimalDelay(float distance_mm, bool is_myelinated = false);
    
    /**
     * @brief Update synaptic plasticity using STDP rule
     */
    void applySTDP(Synapse& synapse, float pre_spike_time, float post_spike_time);
    
    /**
     * @brief Apply homeostatic scaling to maintain network stability
     */
    void applyHomeostaticScaling(std::vector<Synapse*>& synapses, float target_activity);
}

#endif // SYNAPSE_H