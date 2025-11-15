#ifndef MODULAR_NEURAL_NETWORK_H
#define MODULAR_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include <map>

// Forward declarations
class NeuralModule;
struct NetworkConfig;  // FIX: Use struct to match NetworkConfig.h

/**
 * @brief Modular neural network container for managing multiple neural modules
 * 
 * This class serves as a container and coordinator for multiple neural modules,
 * allowing them to work together as a cohesive modular neural network system.
 * It provides module registration, update coordination, and inter-module communication.
 */
class ModularNeuralNetwork {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Default constructor
     */
    ModularNeuralNetwork();
    
    /**
     * @brief Destructor
     */
    virtual ~ModularNeuralNetwork();
    
    /**
     * @brief Initialize the modular network
     * @return Success status of initialization
     */
    bool initialize();
    
    /**
     * @brief Shutdown the modular network
     */
    void shutdown();
    
    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add a neural module to the network
     * @param module Unique pointer to the neural module
     */
    void add_module(std::unique_ptr<NeuralModule> module);
    
    /**
     * @brief Remove a module by name
     * @param module_name Name of the module to remove
     * @return Success status of removal
     */
    bool remove_module(const std::string& module_name);
    
    /**
     * @brief Get a module by name
     * @param module_name Name of the module
     * @return Pointer to the module, or nullptr if not found
     */
    NeuralModule* get_module(const std::string& module_name) const;
    
    /**
     * @brief Get all module names
     * @return Vector of module names
     */
    std::vector<std::string> get_module_names() const;
    
    /**
     * @brief Get number of registered modules
     * @return Number of modules
     */
    size_t get_module_count() const;
    
    // ========================================================================
    // NETWORK OPERATIONS
    // ========================================================================
    
    /**
     * @brief Update all modules in the network
     * @param dt Time step for the update
     */
    void update(float dt);
    
    /**
     * @brief Update a specific module
     * @param module_name Name of the module to update
     * @param dt Time step for the update
     */
    void update_module(const std::string& module_name, float dt);
    
    /**
     * @brief Process inter-module communications
     */
    void process_inter_module_communication();
    
    /**
     * @brief Reset all modules to initial state
     */
    void reset();
    
    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get performance metrics for all modules
     * @return Map of module names to their performance metrics
     */
    std::map<std::string, std::map<std::string, float>> getPerformanceMetrics() const;
    
    /**
     * @brief Get total network activity
     * @return Average activity across all modules
     */
    float get_total_activity() const;
    
    /**
     * @brief Check if network is stable
     * @return True if all modules are operating within normal parameters
     */
    bool is_stable() const;

private:
    // Module storage
    std::vector<std::unique_ptr<NeuralModule>> modules_;
    std::map<std::string, NeuralModule*> module_map_;
    
    // Network state
    bool is_initialized_;
    bool is_shutdown_;
    
    // Performance tracking
    float total_activity_;
    size_t update_count_;
    
    // Helper methods
    void update_performance_metrics();
    void validate_modules();
};

#endif // MODULAR_NEURAL_NETWORK_H