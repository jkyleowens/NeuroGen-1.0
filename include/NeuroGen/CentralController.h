// ============================================================================
// CENTRAL CONTROLLER HEADER - CORRECTED
// File: include/NeuroGen/CentralController.h
// ============================================================================

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <sstream>

// Forward declarations to avoid circular dependencies
class ControllerModule;
class NeuralModule; 
class AutonomousLearningAgent;
class CognitiveModule;
class MotorModule;

/**
 * @brief Central Controller for Task Automation System
 * 
 * This class serves as the main coordinator for the ANIMA-based
 * task automation simulation, integrating neural modules, screen
 * element processing, and autonomous decision making with a focus
 * on natural language processing capabilities.
 */
class CentralController {
public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    /**
     * @brief Constructor - initializes the central controller
     */
    CentralController();
    
    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~CentralController();
    
    // ========================================================================
    // INITIALIZATION AND LIFECYCLE
    // ========================================================================
    
    /**
     * @brief Initialize all subsystems
     * @return Success status
     */
    bool initialize();
    
    /**
     * @brief Shutdown all systems gracefully
     */
    void shutdown();
    
    // ========================================================================
    // MAIN CONTROL INTERFACE
    // ========================================================================
    
    /**
     * @brief Run cognitive cycles
     * @param cycles Number of cycles to run
     */
    void run(int cycles = 1);
    
    // ========================================================================
    // SYSTEM STATUS AND MONITORING
    // ========================================================================
    
    /**
     * @brief Check if system is initialized
     * @return Initialization status
     */
    bool isInitialized() const { return is_initialized_; }
    
    /**
     * @brief Get detailed system status
     * @return Status string with detailed information
     */
    std::string getSystemStatus() const;
    
    /**
     * @brief Get overall system performance metric
     * @return Performance value [0.0-1.0]
     */
    float getSystemPerformance() const;
    
    /**
     * @brief Get number of completed cycles
     * @return Cycle count
     */
    int getCycleCount() const { return cycle_count_; }
    
private:
    // ========================================================================
    // INTERNAL COMPONENTS
    // ========================================================================
    
    // Core system components
    std::unique_ptr<ControllerModule> neuro_controller_;
    std::unique_ptr<AutonomousLearningAgent> learning_agent_;
    
    // Neural modules for basic cognitive functions
    std::shared_ptr<NeuralModule> perception_module_;
    std::shared_ptr<NeuralModule> planning_module_;
    std::shared_ptr<NeuralModule> motor_module_;
    
    // Task-level modules
    std::shared_ptr<CognitiveModule> cognitive_module_;
    std::shared_ptr<MotorModule> motor_task_module_;
    
    // ========================================================================
    // STATE VARIABLES
    // ========================================================================
    
    // System state
    bool is_initialized_;
    int cycle_count_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    /**
     * @brief Initialize neural modules with appropriate configurations
     */
    void initialize_neural_modules();
    
    /**
     * @brief Initialize task-level modules
     */
    void initialize_task_modules();
    
    /**
     * @brief Execute one cognitive cycle
     */
    void execute_cognitive_cycle();
    
    /**
     * @brief Update and display performance metrics
     */
    void update_performance_metrics();
};