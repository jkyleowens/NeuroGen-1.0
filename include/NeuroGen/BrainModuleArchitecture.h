#ifndef BRAIN_MODULE_ARCHITECTURE_H
#define BRAIN_MODULE_ARCHITECTURE_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <set>
#include <queue>
#include <thread>
#include <condition_variable>

// NeuroGen Framework Includes
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/LearningStateManager.h"

// Forward declarations
class LearningStateManager;
class NetworkCUDA;
class ContinuousLearningAgent;

/**
 * @brief Brain-inspired modular architecture for autonomous learning agent
 * 
 * This class implements the complete brain-inspired modular architecture
 * with proper initialization, dynamic sizing, inter-module connections,
 * persistent learning capabilities, memory consolidation, and attention
 * mechanisms based on biological neural networks.
 * 
 * Key Features:
 * - Variable-size modular architecture with hierarchical processing
 * - Independent state saving/loading for each network module
 * - Advanced synaptic plasticity with STDP, homeostasis, and reward modulation
 * - Structural plasticity enabling dynamic synaptic pruning and growth
 * - Central attention/control mechanisms for context-dependent orchestration
 * - Inter-module learning with Hebbian plasticity
 * - Memory consolidation during downtime
 * - CUDA GPU acceleration support
 * - Biological realism with Dale's principle and realistic firing dynamics
 */
class BrainModuleArchitecture : public std::enable_shared_from_this<BrainModuleArchitecture> {
public:
    // ========================================================================
    // MODULE TYPES (Based on Biological Brain Organization)
    // ========================================================================
    
    // Enumeration for different brain module types
    enum class ModuleType {
        PREFRONTAL_CORTEX,      // Executive functions, decision making, planning
        MOTOR_CORTEX,           // Motor control and execution
        HIPPOCAMPUS,            // Memory formation and retrieval
        AMYGDALA,               // Emotional processing
        THALAMUS,               // Sensory and motor signal relay
        CEREBELLUM,             // Fine motor control and coordination
        REWARD_SYSTEM,          // Reward processing and reinforcement learning
        ATTENTION_SYSTEM,       // Attention and focus control
        WORKING_MEMORY,         // Short-term memory and manipulation
        LANGUAGE_PROCESSING,    // Language comprehension and generation
        DEFAULT                 // General-purpose module
    };
    
    // ========================================================================
    // CONFIGURATION STRUCTURES
    // ========================================================================
    
    /**
     * @brief Configuration for individual neural modules
     */
    struct ModuleConfig {
        ModuleType type;
        std::string name;
        std::string description;
        
        // Network topology
        size_t input_size;
        size_t output_size;
        size_t internal_neurons;
        size_t cortical_columns;
        size_t layers_per_column;
        
        // Learning parameters
        float learning_rate;
        float plasticity_strength;
        float attention_sensitivity;
        float fatigue_resistance;
        
        // Biological features
        bool enable_stdp;                    // Spike-timing dependent plasticity
        bool enable_homeostasis;             // Homeostatic scaling
        bool enable_structural_plasticity;   // Dynamic synapse formation/removal
        bool enable_learning_persistence;    // State saving/loading
        bool enable_neuromodulation;        // Dopamine/acetylcholine/norepinephrine
        
        // Module connectivity
        std::vector<std::string> input_connections;   // Source modules
        std::vector<std::string> output_connections;  // Target modules
        std::vector<std::string> lateral_connections; // Same-level modules
        
        // Learning persistence parameters
        float consolidation_strength = 0.1f;         // Memory consolidation rate
        float eligibility_decay_rate = 0.95f;        // Eligibility trace decay
        float synaptic_tag_threshold = 0.1f;         // Threshold for synaptic tagging
        float memory_capacity_factor = 1.0f;         // Relative memory capacity
        
        // Attention and control
        float baseline_activity = 0.1f;              // Resting activity level
        float max_activity = 2.0f;                   // Maximum activity level
        float attention_decay_rate = 0.98f;          // Attention decay over time
        
        ModuleConfig() : type(ModuleType::PREFRONTAL_CORTEX), input_size(1000), output_size(500),
                        internal_neurons(1500), cortical_columns(16), layers_per_column(6),
                        learning_rate(0.001f), plasticity_strength(1.0f), attention_sensitivity(0.7f),
                        fatigue_resistance(0.8f), enable_stdp(true), enable_homeostasis(true),
                        enable_structural_plasticity(true), enable_learning_persistence(true),
                        enable_neuromodulation(true) {}
    };
    
    /**
     * @brief Inter-module connection specification
     */
    struct InterModuleConnection {
        std::string source_module;
        std::string source_port;
        std::string target_module;
        std::string target_port;
        
        // Connection properties
        float connection_strength;              // Current strength (0-1)
        float base_strength;                   // Initial/baseline strength
        float max_strength;                    // Maximum allowed strength
        bool plastic;                          // Whether connection can change
        bool bidirectional;                    // Bidirectional connection
        size_t connection_size;                // Number of actual connections
        float delay_ms;                        // Transmission delay
        
        // Learning parameters
        float plasticity_rate = 0.001f;       // Learning rate for this connection
        float hebbian_strength = 0.1f;        // Hebbian learning coefficient
        float anti_hebbian_strength = 0.05f;  // Anti-Hebbian (forgetting) coefficient
        float homeostatic_target = 0.1f;      // Target activity level
        
        // Connection type and function
        std::string connection_type;           // "excitatory", "inhibitory", "modulatory"
        std::string function_type;             // "feedforward", "feedback", "lateral"
        
        InterModuleConnection() : connection_strength(0.1f), base_strength(0.1f), max_strength(1.0f),
                                plastic(true), bidirectional(false), connection_size(100), delay_ms(1.0f) {}
    };
    
    /**
     * @brief Global architecture configuration
     */
    struct ArchitectureConfig {
        // Global learning parameters
        float global_learning_rate_multiplier;
        float global_plasticity_modifier;
        float global_attention_strength;
        
        // Memory and consolidation
        bool enable_global_consolidation;
        int consolidation_interval_minutes;
        float consolidation_threshold;
        
        // Neuromodulation
        float dopamine_baseline;
        float acetylcholine_baseline;
        float norepinephrine_baseline;
        float serotonin_baseline;
        
        // Structural plasticity
        bool enable_global_pruning;
        float pruning_threshold;
        float growth_probability;
        
        // Performance monitoring
        bool enable_performance_tracking;
        size_t performance_history_size;
        float stability_threshold;

        ArchitectureConfig()
            : global_learning_rate_multiplier(1.0f),
              global_plasticity_modifier(1.0f),
              global_attention_strength(1.0f),
              enable_global_consolidation(true),
              consolidation_interval_minutes(30),
              consolidation_threshold(0.5f),
              dopamine_baseline(0.1f),
              acetylcholine_baseline(0.1f),
              norepinephrine_baseline(0.1f),
              serotonin_baseline(0.1f),
              enable_global_pruning(true),
              pruning_threshold(0.01f),
              growth_probability(0.001f),
              enable_performance_tracking(true),
              performance_history_size(10000),
              stability_threshold(0.1f) {}
    };
    
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor with optional configuration
     * @param config Global architecture configuration
     */
    explicit BrainModuleArchitecture(const ArchitectureConfig& config = ArchitectureConfig{});
    
    /**
     * @brief Destructor with proper cleanup
     */
    virtual ~BrainModuleArchitecture();
    
    /**
     * @brief Initialize the architecture with default brain modules
     * @param visual_input_width Width of visual input
     * @param visual_input_height Height of visual input
     * @return Success status
     */
    bool initialize(int visual_input_width = 128, int visual_input_height = 128);
    
    /**
     * @brief Initialize with custom module configurations
     * @param module_configs Vector of module configurations
     * @param connections Vector of inter-module connections
     * @return Success status
     */
    bool initializeCustom(const std::vector<ModuleConfig>& module_configs,
                         const std::vector<InterModuleConnection>& connections);
    
    /**
     * @brief Validate architecture configuration
     * @return Validation success status with error details
     */
    std::pair<bool, std::string> validateConfiguration() const;
    
    /**
     * @brief Reset architecture to initial state
     * @param preserve_structure If true, keep module structure but reset learning
     */
    void reset(bool preserve_structure = true);
    
    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add a new module to the architecture
     * @param config Module configuration
     * @return Success status with error message if failed
     */
    std::pair<bool, std::string> addModule(const ModuleConfig& config);
    
    /**
     * @brief Remove a module from the architecture
     * @param module_name Name of module to remove
     * @param cleanup_connections Whether to remove associated connections
     * @return Success status
     */
    bool removeModule(const std::string& module_name, bool cleanup_connections = true);
    
    /**
     * @brief Update module configuration
     * @param module_name Name of the module
     * @param config New configuration
     * @return Success status
     */
    bool updateModuleConfig(const std::string& module_name, const ModuleConfig& config);
    
    /**
     * @brief Get module by name
     * @param module_name Name of the module
     * @return Shared pointer to module (nullptr if not found)
     */
    std::shared_ptr<EnhancedNeuralModule> getModule(const std::string& module_name) const;
    
    /**
     * @brief Get list of all module names
     * @return Vector of module names sorted alphabetically
     */
    std::vector<std::string> getModuleNames() const;
    
    /**
     * @brief Get modules by type
     * @param type Module type to filter by
     * @return Vector of module names of the specified type
     */
    std::vector<std::string> getModulesByType(ModuleType type) const;
    
    /**
     * @brief Get module configuration
     * @param module_name Name of the module
     * @return Module configuration (empty if not found)
     */
    ModuleConfig getModuleConfig(const std::string& module_name) const;
    
    /**
     * @brief Check if module exists
     * @param module_name Name of the module
     * @return True if module exists
     */
    bool hasModule(const std::string& module_name) const;
    
    /**
     * @brief Get module count
     * @return Number of modules in the architecture
     */
    size_t getModuleCount() const;
    
    // ========================================================================
    // INTER-MODULE CONNECTION MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add connection between modules
     * @param connection Connection specification
     * @return Success status with error message if failed
     */
    std::pair<bool, std::string> addConnection(const InterModuleConnection& connection);
    
    /**
     * @brief Remove connection between modules
     * @param source_module Source module name
     * @param target_module Target module name
     * @return Success status
     */
    bool removeConnection(const std::string& source_module, const std::string& target_module);
    
    /**
     * @brief Remove all connections for a module
     * @param module_name Module name
     * @return Number of connections removed
     */
    size_t removeAllConnections(const std::string& module_name);
    
    /**
     * @brief Update inter-module connection strengths with learning
     * @param learning_rate_multiplier Global learning rate modifier
     */
    void updateInterModuleConnections(float learning_rate_multiplier = 1.0f);
    
    /**
     * @brief Get connection strength between modules
     * @param source_module Source module name
     * @param target_module Target module name
     * @return Connection strength (0-1, -1 if connection doesn't exist)
     */
    float getConnectionStrength(const std::string& source_module, const std::string& target_module) const;
    
    /**
     * @brief Set connection strength between modules
     * @param source_module Source module name
     * @param target_module Target module name
     * @param strength New connection strength (0-1)
     * @return Success status
     */
    bool setConnectionStrength(const std::string& source_module, const std::string& target_module, float strength);
    
    /**
     * @brief Get all inter-module connections
     * @return Vector of connection configurations
     */
    std::vector<InterModuleConnection> getConnections() const;
    
    /**
     * @brief Get connections for a specific module
     * @param module_name Module name
     * @param incoming If true, get incoming connections; if false, get outgoing
     * @return Vector of connections
     */
    std::vector<InterModuleConnection> getModuleConnections(const std::string& module_name, bool incoming = true) const;
    
    /**
     * @brief Check if connection exists
     * @param source_module Source module name
     * @param target_module Target module name
     * @return True if connection exists
     */
    bool hasConnection(const std::string& source_module, const std::string& target_module) const;
    
    // ========================================================================
    // CORE PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process input through the entire architecture
     * @param inputs Input data vector
     * @return Map of module names to their output vectors
     */
    std::map<std::string, std::vector<float>> processInput(const std::vector<float>& inputs);
    
    /**
     * @brief Process input with learning updates
     * @param inputs Input data
     * @param reward Global reward signal
     * @param novelty_signal Novelty/surprise signal
     * @return Map of module outputs
     */
    std::map<std::string, std::vector<float>> processInputWithLearning(
        const std::vector<float>& inputs, 
        float reward = 0.0f, 
        float novelty_signal = 0.0f);
    
    /**
     * @brief Process multi-modal input
     * @param visual_input Visual data
     * @param text_input Text/language data
     * @param context_input Additional context
     * @param reward Reward signal
     * @return Combined output from all modules
     */
    std::map<std::string, std::vector<float>> processMultiModalInput(
        const std::vector<float>& visual_input,
        const std::vector<float>& text_input,
        const std::vector<float>& context_input,
        float reward = 0.0f);
    
    /**
     * @brief Update all modules with time step
     * @param dt Time step in seconds
     * @param global_reward Global reward signal
     */
    void update(float dt, float global_reward = 0.0f);
    
    /**
     * @brief Get outputs from specific module
     * @param module_name Module name
     * @return Output vector (empty if module not found)
     */
    std::vector<float> getModuleOutput(const std::string& module_name) const;
    
    /**
     * @brief Get combined output from multiple modules
     * @param module_names Vector of module names
     * @return Concatenated output vector
     */
    std::vector<float> getCombinedOutput(const std::vector<std::string>& module_names) const;
    
    // ========================================================================
    // ATTENTION AND CONTROL MECHANISMS
    // ========================================================================
    
    /**
     * @brief Update attention weights for all modules
     * @param attention_weights Vector of attention weights per module
     */
    void updateAttention(const std::vector<float>& attention_weights);
    
    /**
     * @brief Update attention weights by module name
     * @param attention_map Map of module names to attention weights
     */
    void updateAttention(const std::map<std::string, float>& attention_map);
    
    /**
     * @brief Update attention based on context vector
     * @param context_vector Current context information
     */
    void updateGlobalAttention(const std::vector<float>& context_vector);
    
    /**
     * @brief Apply learning signal to all modules
     * @param reward Global reward signal
     * @param prediction_error Prediction error signal
     */
    void applyLearning(float reward, float prediction_error);
    
    /**
     * @brief Apply neuromodulation to all modules
     * @param dopamine Dopamine level (reward/motivation)
     * @param acetylcholine Acetylcholine level (attention/learning)
     * @param norepinephrine Norepinephrine level (arousal/stress)
     * @param serotonin Serotonin level (mood/regulation)
     */
    void applyNeuromodulation(float dopamine, float acetylcholine, float norepinephrine, float serotonin = 0.1f);
    
    /**
     * @brief Get current attention weights
     * @return Map of module names to attention weights
     */
    std::map<std::string, float> getAttentionWeights() const;
    
    /**
     * @brief Get attention distribution as normalized probabilities
     * @return Map of module names to attention probabilities (sum = 1.0)
     */
    std::map<std::string, float> getAttentionDistribution() const;
    
    /**
     * @brief Focus attention on specific modules
     * @param module_names Modules to focus attention on
     * @param focus_strength Strength of focus (0-1)
     */
    void focusAttention(const std::vector<std::string>& module_names, float focus_strength = 0.8f);
    
    /**
     * @brief Get total network activity
     * @return Average activity across all modules
     */
    float getTotalActivity() const;
    
    /**
     * @brief Get activity for specific module
     * @param module_name Module name
     * @return Module activity level
     */
    float getModuleActivity(const std::string& module_name) const;
    
    // ========================================================================
    // LEARNING STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Get learning state for all modules
     * @return Complete session learning state
     */
    SessionLearningState getGlobalLearningState() const;
    
    /**
     * @brief Apply learning state to all modules
     * @param state Session learning state to apply
     * @return Success status with error message if failed
     */
    std::pair<bool, std::string> applyGlobalLearningState(const SessionLearningState& state);
    
    /**
     * @brief Save learning state to directory
     * @param save_directory Directory to save state files
     * @param checkpoint_name Name for this checkpoint
     * @return Success status
     */
    bool saveLearningState(const std::string& save_directory, const std::string& checkpoint_name);
    
    /**
     * @brief Load learning state from directory
     * @param save_directory Directory containing state files
     * @param checkpoint_name Name of checkpoint to load (latest if empty)
     * @return Success status
     */
    bool loadLearningState(const std::string& save_directory, const std::string& checkpoint_name = "");
    
    /**
     * @brief Get available checkpoints in directory
     * @param save_directory Directory to search
     * @return Vector of checkpoint names with timestamps
     */
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> 
        getAvailableCheckpoints(const std::string& save_directory) const;
    
    /**
     * @brief Get inter-module connection state
     * @return Vector of all inter-module connections with learning state
     */
    std::vector<InterModuleConnectionState> getInterModuleConnectionState() const;
    
    /**
     * @brief Apply inter-module connection state
     * @param connections Vector of connection states to apply
     * @return Success status
     */
    bool applyInterModuleConnectionState(const std::vector<InterModuleConnectionState>& connections);
    
    /**
     * @brief Create backup of current state
     * @param backup_directory Directory for backup
     * @param backup_name Name for backup
     * @return Success status
     */
    bool createBackup(const std::string& backup_directory, const std::string& backup_name) const;
    
    /**
     * @brief Restore from backup
     * @param backup_directory Directory containing backup
     * @param backup_name Name of backup to restore
     * @return Success status
     */
    bool restoreFromBackup(const std::string& backup_directory, const std::string& backup_name);
    
    // ========================================================================
    // MEMORY CONSOLIDATION
    // ========================================================================
    
    /**
     * @brief Perform memory consolidation across all modules
     * @param consolidation_strength Consolidation strength (0-1)
     * @return Total number of synapses consolidated
     */
    size_t performGlobalMemoryConsolidation(float consolidation_strength = 0.1f);
    
    /**
     * @brief Consolidate memories in specific module
     * @param module_name Module to consolidate
     * @param consolidation_strength Strength of consolidation
     * @return Number of synapses consolidated
     */
    size_t consolidateModule(const std::string& module_name, float consolidation_strength);
    
    /**
     * @brief Consolidate memories in modules of specific type
     * @param type Module type to consolidate
     * @param consolidation_strength Strength of consolidation
     * @return Total number of synapses consolidated
     */
    size_t consolidateModuleType(ModuleType type, float consolidation_strength);
    
    /**
     * @brief Schedule automatic memory consolidation
     * @param enable Enable automatic consolidation
     * @param interval_minutes Interval between consolidations
     */
    void enableAutoConsolidation(bool enable, int interval_minutes = 30);
    
    /**
     * @brief Check if consolidation is needed
     * @return True if consolidation should be performed
     */
    bool shouldConsolidate() const;
    
    /**
     * @brief Get time since last consolidation
     * @return Minutes since last consolidation
     */
    int getMinutesSinceConsolidation() const;
    
    // ========================================================================
    // STRUCTURAL PLASTICITY
    // ========================================================================
    
    /**
     * @brief Perform structural plasticity updates
     * @param growth_rate Rate of new synapse formation
     * @param pruning_rate Rate of weak synapse removal
     */
    void updateStructuralPlasticity(float growth_rate = 0.001f, float pruning_rate = 0.002f);
    
    /**
     * @brief Prune weak connections throughout architecture
     * @param pruning_threshold Threshold for connection removal
     * @return Number of connections pruned
     */
    size_t pruneWeakConnections(float pruning_threshold = 0.01f);
    
    /**
     * @brief Grow new connections based on activity
     * @param growth_probability Probability of new connection formation
     * @return Number of new connections formed
     */
    size_t growNewConnections(float growth_probability = 0.001f);
    
    /**
     * @brief Optimize network topology
     * @param optimization_strength Strength of optimization (0-1)
     * @return Improvement score
     */
    float optimizeTopology(float optimization_strength = 0.1f);
    
    // ========================================================================
    // PERFORMANCE MONITORING AND STATISTICS
    // ========================================================================
    
    /**
     * @brief Get architecture performance metrics
     * @return Performance metrics for all modules and connections
     */
    std::map<std::string, float> getPerformanceMetrics() const;
    
    /**
     * @brief Get detailed learning statistics
     * @return Learning metrics for all modules
     */
    std::map<std::string, std::map<std::string, float>> getLearningStats() const;
    
    /**
     * @brief Get performance history for module
     * @param module_name Module name
     * @param window_size Number of recent samples to include
     * @return Vector of performance values over time
     */
    std::vector<float> getModulePerformanceHistory(const std::string& module_name, size_t window_size = 100) const;
    
    /**
     * @brief Calculate global prediction error
     * @param expected_outputs Expected outputs for each module
     * @return Global prediction error
     */
    float calculateGlobalPredictionError(const std::map<std::string, std::vector<float>>& expected_outputs) const;
    
    /**
     * @brief Check if architecture is stable
     * @return Stability status with details
     */
    std::pair<bool, std::string> isStable() const;
    
    /**
     * @brief Get architecture statistics
     * @return Map of architecture statistics
     */
    std::map<std::string, uint32_t> getArchitectureStatistics() const;
    
    /**
     * @brief Get detailed performance report
     * @return JSON-formatted performance report
     */
    std::string getPerformanceReport() const;
    
    /**
     * @brief Check if learning is progressing
     * @param window_size Number of recent steps to analyze
     * @return True if learning progress is detected
     */
    bool isLearningProgressing(size_t window_size = 1000) const;
    
    // ========================================================================
    // ARCHITECTURE COMPATIBILITY AND VALIDATION
    // ========================================================================
    
    /**
     * @brief Calculate architecture hash for compatibility checking
     * @return Hash string representing current architecture
     */
    std::string calculateArchitectureHash() const;
    
    /**
     * @brief Validate architecture compatibility with saved state
     * @param state_hash Hash from saved state
     * @return Compatibility status with detailed explanation
     */
    std::pair<bool, std::string> validateArchitectureCompatibility(const std::string& state_hash) const;
    
    /**
     * @brief Get detailed architecture description
     * @return JSON-formatted architecture description
     */
    std::string getArchitectureDescription() const;
    
    /**
     * @brief Compare with another architecture
     * @param other Other architecture to compare with
     * @return Similarity score (0-1) and differences
     */
    std::pair<float, std::vector<std::string>> compareArchitecture(const BrainModuleArchitecture& other) const;
    
    /**
     * @brief Validate internal consistency
     * @return Validation results with any issues found
     */
    std::pair<bool, std::vector<std::string>> validateInternalConsistency() const;
    
    // ========================================================================
    // CUDA INTEGRATION
    // ========================================================================
    
    /**
     * @brief Set CUDA network interface
     * @param cuda_network Pointer to CUDA network implementation
     */
    void setCUDANetwork(std::shared_ptr<NetworkCUDA> cuda_network);
    
    /**
     * @brief Enable GPU acceleration
     * @param enable Enable GPU processing
     * @return Success status with error message if failed
     */
    std::pair<bool, std::string> enableGPUAcceleration(bool enable);
    
    /**
     * @brief Check if GPU acceleration is available
     * @return True if GPU is available and enabled
     */
    bool isGPUEnabled() const;
    
    /**
     * @brief Get GPU memory usage
     * @return Memory usage statistics
     */
    std::map<std::string, size_t> getGPUMemoryUsage() const;
    
    /**
     * @brief Synchronize with GPU
     * @param force_sync Force synchronization even if not needed
     */
    void synchronizeGPU(bool force_sync = false);
    
    // ========================================================================
    // ADVANCED FEATURES
    // ========================================================================
    
    /**
     * @brief Enable meta-learning capabilities
     * @param enable Enable meta-learning
     * @return Success status
     */
    bool enableMetaLearning(bool enable);
    
    /**
     * @brief Set learning state manager
     * @param manager Shared pointer to learning state manager
     */
    void setLearningStateManager(std::shared_ptr<LearningStateManager> manager);
    
    /**
     * @brief Set continuous learning agent reference
     * @param agent Weak pointer to continuous learning agent
     */
    void setContinuousLearningAgent(std::weak_ptr<ContinuousLearningAgent> agent);
    
    /**
     * @brief Get architecture configuration
     * @return Current architecture configuration
     */
    ArchitectureConfig getArchitectureConfig() const;
    
    /**
     * @brief Update architecture configuration
     * @param config New configuration
     * @return Success status
     */
    bool updateArchitectureConfig(const ArchitectureConfig& config);

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Core configuration
    ArchitectureConfig architecture_config_;
    
    // Core modular network
    std::unique_ptr<ModularNeuralNetwork> modular_network_;
    
    // Module management
    std::map<std::string, std::shared_ptr<EnhancedNeuralModule>> modules_;
    std::map<std::string, ModuleConfig> module_configs_;
    std::vector<InterModuleConnection> connections_;
    
    // Inter-module connection tracking
    std::map<std::pair<std::string, std::string>, InterModuleConnectionState> inter_module_connections_;
    std::map<std::pair<std::string, std::string>, float> connection_usage_history_;
    std::map<std::pair<std::string, std::string>, std::vector<float>> connection_activity_history_;
    
    // Attention and control
    std::map<std::string, float> attention_weights_;
    std::map<std::string, float> attention_history_;
    std::vector<float> global_context_vector_;
    float global_inhibition_level_ = 0.1f;
    
    // Neuromodulation
    float global_dopamine_level_ = 0.1f;
    float global_acetylcholine_level_ = 0.1f;
    float global_norepinephrine_level_ = 0.1f;
    float global_serotonin_level_ = 0.1f;
    
    // Learning session information
    mutable std::mutex learning_state_mutex_;
    uint64_t global_learning_steps_ = 0;
    float global_reward_accumulator_ = 0.0f;
    std::chrono::steady_clock::time_point last_update_time_;
    std::chrono::steady_clock::time_point creation_time_;
    
    // Performance tracking
    std::map<std::string, std::vector<float>> module_performance_history_;
    std::map<std::string, float> module_prediction_errors_;
    std::map<std::string, float> module_stability_scores_;
    size_t performance_history_size_ = 10000;
    
    // Dynamic sizing parameters
    int visual_input_width_ = 128;
    int visual_input_height_ = 128;
    size_t visual_feature_size_ = 16384; // 128*128
    
    // External interfaces
    std::shared_ptr<NetworkCUDA> cuda_network_;
    std::shared_ptr<LearningStateManager> learning_state_manager_;
    std::weak_ptr<ContinuousLearningAgent> continuous_learning_agent_;
    bool gpu_enabled_ = false;
    bool meta_learning_enabled_ = false;
    
    // Auto-consolidation
    std::atomic<bool> auto_consolidation_enabled_{false};
    std::atomic<int> auto_consolidation_interval_{30};
    std::chrono::steady_clock::time_point last_consolidation_;
    
    // Background processing
    std::atomic<bool> background_processing_enabled_{false};
    std::thread background_thread_;
    std::condition_variable background_cv_;
    std::mutex background_mutex_;
    
    // Thread safety
    mutable std::mutex modules_mutex_;
    mutable std::mutex connections_mutex_;
    mutable std::mutex performance_mutex_;
    mutable std::mutex attention_mutex_;
    mutable std::mutex neuromodulation_mutex_;
    
    // Validation and debugging
    mutable std::atomic<bool> validation_enabled_{true};
    mutable std::vector<std::string> last_validation_errors_;
    
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize default brain modules
     * @return Success status
     */
    bool initializeDefaultModules();
    
    /**
     * @brief Initialize inter-module connections
     */
    void initializeInterModuleConnections();
    
    /**
     * @brief Create default module configuration
     * @param type Module type
     * @param name Module name
     * @return Default configuration for the module type
     */
    ModuleConfig createDefaultModuleConfig(ModuleType type, const std::string& name);
    
    /**
     * @brief Update connection usage statistics
     * @param source_module Source module name
     * @param target_module Target module name
     * @param activation_strength Strength of activation
     */
    void updateConnectionUsage(const std::string& source_module, 
                              const std::string& target_module, 
                              float activation_strength);
    
    /**
     * @brief Apply Hebbian learning for inter-module connections
     * @param source_activity Source module activity
     * @param target_activity Target module activity
     * @param connection Inter-module connection to update
     */
    void applyHebbianLearning(float source_activity, float target_activity, 
                             InterModuleConnectionState& connection);
    
    /**
     * @brief Route data between modules
     * @param source_module Source module name
     * @param target_module Target module name
     * @param data Data to route
     * @return Processed data after routing
     */
    std::vector<float> routeData(const std::string& source_module, 
                                const std::string& target_module, 
                                const std::vector<float>& data);
    
    /**
     * @brief Calculate module attention based on activity and context
     * @param module_name Module name
     * @param context_vector Current context
     * @return Calculated attention weight
     */
    float calculateModuleAttention(const std::string& module_name, 
                                  const std::vector<float>& context_vector);
    
    /**
     * @brief Save individual module learning state
     * @param module_name Module name
     * @param save_directory Directory to save to
     * @return Success status
     */
    bool saveModuleLearningState(const std::string& module_name, const std::string& save_directory) const;
    
    /**
     * @brief Load individual module learning state
     * @param module_name Module name
     * @param save_directory Directory to load from
     * @return Success status
     */
    bool loadModuleLearningState(const std::string& module_name, const std::string& save_directory);
    
    /**
     * @brief Update performance metrics for all modules
     */
    void updatePerformanceMetrics();
    
    /**
     * @brief Validate module connections
     * @return True if all connections are valid
     */
    bool validateConnections() const;
    
    /**
     * @brief Background processing thread function
     */
    void backgroundProcessingLoop();
    
    /**
     * @brief Cleanup resources and connections
     */
    void cleanup();
    
    /**
     * @brief Get module type string representation
     * @param type Module type
     * @return String representation
     */
    static std::string moduleTypeToString(ModuleType type);
    
    /**
     * @brief Parse module type from string
     * @param type_str String representation
     * @return Module type
     */
    static ModuleType stringToModuleType(const std::string& type_str);
    
    /**
     * @brief Generate unique module name
     * @param base_name Base name for the module
     * @return Unique module name
     */
    std::string generateUniqueModuleName(const std::string& base_name) const;
    
    /**
     * @brief Validate module configuration
     * @param config Module configuration to validate
     * @return Validation result with error details
     */
    std::pair<bool, std::string> validateModuleConfig(const ModuleConfig& config) const;
    
    /**
     * @brief Validate connection configuration
     * @param connection Connection configuration to validate
     * @return Validation result with error details
     */
    std::pair<bool, std::string> validateConnectionConfig(const InterModuleConnection& connection) const;
};

#endif // BRAIN_MODULE_ARCHITECTURE_H