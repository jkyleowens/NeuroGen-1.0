#ifndef DYNAMIC_NEURAL_NETWORK_H
#define DYNAMIC_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <NeuroGen/NetworkStats.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NeuralConstants.h>

/**
 * @brief Dynamic Neural Network with Breakthrough Biological Features
 * 
 * This represents the pinnacle of our neuromorphic computing research,
 * implementing a fully dynamic neural network that exhibits emergent
 * intelligence through sophisticated biological mechanisms including:
 * - Activity-dependent neurogenesis and pruning
 * - Multi-timescale synaptic plasticity
 * - Homeostatic regulation and adaptation
 * - Neuromodulatory systems (dopamine, acetylcholine, serotonin)
 * - Critical dynamics and self-organization
 * - Real-time learning and memory consolidation
 */
class DynamicNeuralNetwork {
private:
    // Network topology
    int initial_neurons_;
    int initial_synapses_;
    int max_neurons_;
    int max_synapses_;
    std::atomic<int> current_neurons_;
    std::atomic<int> current_synapses_;
    
    // Dynamic structure management
    std::vector<bool> active_neurons_;
    std::vector<bool> active_synapses_;
    std::vector<float> neuron_health_;
    std::vector<float> synapse_strength_;
    
    // Biological state variables
    std::vector<float> membrane_potentials_;
    std::vector<float> calcium_concentrations_;
    std::vector<float> dopamine_levels_;
    std::vector<float> acetylcholine_levels_;
    std::vector<float> firing_rates_;
    
    // Plasticity and learning
    std::vector<float> synaptic_weights_;
    std::vector<float> eligibility_traces_;
    std::vector<float> plasticity_thresholds_;
    std::vector<int> synapse_ages_;
    
    // Homeostatic regulation
    float target_activity_;
    float activity_tolerance_;
    float regulation_strength_;
    bool homeostatic_enabled_;
    
    // Neuromodulation
    float global_dopamine_;
    float global_acetylcholine_;
    float global_serotonin_;
    float global_norepinephrine_;
    
    // Learning parameters
    float learning_rate_;
    float curiosity_strength_;
    bool learning_rate_adaptation_;
    bool curiosity_driven_learning_;
    
    // Performance metrics
    std::atomic<float> network_efficiency_;
    std::atomic<float> learning_progress_;
    std::atomic<float> adaptation_rate_;
    
    // Thread safety
    mutable std::mutex network_mutex_;
    bool is_initialized_;

public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct dynamic neural network with capacity limits
     * @param initial_neurons Starting number of neurons
     * @param initial_synapses Starting number of synapses
     * @param max_neurons Maximum neuron capacity
     * @param max_synapses Maximum synapse capacity
     */
    DynamicNeuralNetwork(int initial_neurons, int initial_synapses, 
                        int max_neurons, int max_synapses);
    
    /**
     * @brief Virtual destructor for cleanup
     */
    virtual ~DynamicNeuralNetwork();
    
    /**
     * @brief Initialize the dynamic network with biological parameters
     * @return Success status of initialization
     */
    bool initialize();
    
    /**
     * @brief Initialize Phase 3 systems (advanced features)
     * @return Success status of Phase 3 initialization
     */
    bool initializePhase3Systems();
    
    /**
     * @brief Validate network initialization and integrity
     * @return Validation success status
     */
    bool validateInitialization() const;
    
    // ========================================================================
    // MAIN SIMULATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Execute single simulation step with full dynamics
     * @param dt Time step for integration (milliseconds)
     * @param current_time Current simulation time
     */
    void simulationStep(float dt, float current_time);
    
    /**
     * @brief Execute adaptive timestep simulation
     * @param target_dt Target time step
     * @param current_time Current simulation time
     */
    void adaptiveSimulationStep(float target_dt, float current_time);
    
    /**
     * @brief Execute multiple simulation steps efficiently
     * @param num_steps Number of steps to execute
     * @param dt Time step per step
     * @param start_time Starting simulation time
     */
    void multiStepSimulation(int num_steps, float dt, float start_time);
    
    // ========================================================================
    // LEARNING AND ADAPTATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Set global reward signal for reinforcement learning
     * @param reward Reward magnitude [-1.0, 1.0]
     */
    void setRewardSignal(float reward);
    
    /**
     * @brief Set environmental features for adaptive learning
     * @param features Vector of environmental feature values
     */
    void setEnvironmentalFeatures(const std::vector<float>& features);
    
    /**
     * @brief Enable curiosity-driven learning mechanisms
     * @param enable Curiosity learning enable flag
     */
    void enableCuriosityDrivenLearning(bool enable);
    
    /**
     * @brief Set learning rate adaptation behavior
     * @param enable Adaptive learning rate enable flag
     */
    void setLearningRateAdaptation(bool enable);
    
    /**
     * @brief Configure learning goals for directed adaptation
     * @param goals Vector of learning objective values
     */
    void setLearningGoals(const std::vector<float>& goals);
    
    // ========================================================================
    // DYNAMIC STRUCTURE CONTROL
    // ========================================================================
    
    /**
     * @brief Enable neurogenesis (new neuron creation)
     * @param enable Neurogenesis enable flag
     */
    void enableNeurogenesis(bool enable);
    
    /**
     * @brief Enable synaptogenesis (new synapse formation)
     * @param enable Synaptogenesis enable flag
     */
    void enableSynaptogenesis(bool enable);
    
    /**
     * @brief Enable neural pruning (removal of weak connections)
     * @param enable Pruning enable flag
     */
    void enablePruning(bool enable);
    
    /**
     * @brief Set structural plasticity rate
     * @param rate Rate of structural changes [0.0, 1.0]
     */
    void setStructuralPlasticityRate(float rate);
    
    // ========================================================================
    // HOMEOSTATIC CONTROL INTERFACE
    // ========================================================================
    
    /**
     * @brief Set homeostatic regulation targets
     * @param activity_target Target activity level (Hz)
     * @param connectivity_target Target connectivity density
     */
    void setHomeostaticTargets(float activity_target, float connectivity_target);
    
    /**
     * @brief Enable homeostatic regulation mechanisms
     * @param enable Homeostatic regulation enable flag
     */
    void enableHomeostaticRegulation(bool enable);
    
    /**
     * @brief Set stability protection for critical periods
     * @param enable Stability protection enable flag
     */
    void setStabilityProtection(bool enable);
    
    // ========================================================================
    // MONITORING AND ANALYSIS INTERFACE
    // ========================================================================
    
    /**
     * @brief Get current system state
     * @return SystemState structure with current metrics
     */
    SystemState getSystemState() const;
    
    /**
     * @brief Get detailed network statistics
     * @param stats Output vector for comprehensive statistics
     */
    void getDetailedStatistics(std::vector<float>& stats) const;
    
    /**
     * @brief Export network structure to file
     * @param filename Output filename for structure data
     */
    void exportNetworkStructure(const std::string& filename) const;
    
    /**
     * @brief Generate comprehensive performance report
     * @param filename Output filename for performance report
     */
    void generatePerformanceReport(const std::string& filename) const;
    
    // ========================================================================
    // PERFORMANCE OPTIMIZATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Enable adaptive timestep control
     * @param enable Adaptive timestep enable flag
     */
    void enableAdaptiveTimestep(bool enable);
    
    /**
     * @brief Set computation budget for real-time operation
     * @param budget_ms Maximum computation time per step (milliseconds)
     */
    void setComputationBudget(float budget_ms);
    
    /**
     * @brief Enable memory optimization for large networks
     * @param enable Memory optimization enable flag
     */
    void enableMemoryOptimization(bool enable);
    
    /**
     * @brief Optimize network for low latency
     */
    void optimizeForLatency();
    
    /**
     * @brief Optimize network for high throughput
     */
    void optimizeForThroughput();

private:
    // ========================================================================
    // INTERNAL DYNAMICS METHODS
    // ========================================================================
    
    /**
     * @brief Update membrane dynamics for all neurons
     * @param dt Time step
     */
    void updateMembraneDynamics(float dt);
    
    /**
     * @brief Update synaptic transmission
     * @param dt Time step
     */
    void updateSynapticTransmission(float dt);
    
    /**
     * @brief Update calcium dynamics
     * @param dt Time step
     */
    void updateCalciumDynamics(float dt);
    
    /**
     * @brief Update neuromodulator levels
     * @param dt Time step
     */
    void updateNeuromodulation(float dt);
    
    /**
     * @brief Execute synaptic plasticity updates
     * @param dt Time step
     */
    void updateSynapticPlasticity(float dt);
    
    /**
     * @brief Execute homeostatic regulation
     * @param dt Time step
     */
    void executeHomeostaticRegulation(float dt);
    
    /**
     * @brief Execute structural plasticity
     * @param dt Time step
     */
    void executeStructuralPlasticity(float dt);
    
    /**
     * @brief Update network performance metrics
     */
    void updatePerformanceMetrics();
    
    /**
     * @brief Validate network integrity
     * @return Integrity check success
     */
    bool validateNetworkIntegrity() const;
};


#endif // DYNAMIC_NEURAL_NETWORK_H