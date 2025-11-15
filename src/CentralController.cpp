// ============================================================================
// CENTRAL CONTROLLER IMPLEMENTATION - FIXED
// File: src/CentralController.cpp
// ============================================================================

#include "NeuroGen/CentralController.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <iomanip>

// ============================================================================
// CONSTRUCTOR AND DESTRUCTOR
// ============================================================================

CentralController::CentralController() 
    : is_initialized_(false), cycle_count_(0) {
    std::cout << "CentralController: Initializing central coordination system..." << std::endl;
}

CentralController::~CentralController() {
    shutdown();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

bool CentralController::initialize() {
    if (is_initialized_) {
        std::cout << "CentralController: Already initialized" << std::endl;
        return true;
    }
    
    try {
        std::cout << "CentralController: Starting initialization sequence..." << std::endl;
        
        // Step 1: Initialize neural modules
        initialize_neural_modules();
        
        // Step 2: Initialize controller module with proper configuration
        ControllerConfig controller_config;
        controller_config.initial_dopamine_level = 0.4f;
        controller_config.reward_learning_rate = 0.02f;
        controller_config.enable_detailed_logging = true;
        
        neuro_controller_ = std::make_unique<ControllerModule>(controller_config);
        
        // Register neural modules with controller
        neuro_controller_->register_module("PerceptionNet", perception_module_);
        neuro_controller_->register_module("PlanningNet", planning_module_);
        neuro_controller_->register_module("MotorControlNet", motor_module_);
        
        // Step 3: Initialize task modules
        initialize_task_modules();
        
        // Step 4: Initialize autonomous learning agent with NetworkConfig
        // The AutonomousLearningAgent constructor takes NetworkConfig, not ArchitectureConfig
        NetworkConfig agent_config;
        agent_config.num_neurons = 10000;         // Base neuron count
        agent_config.enable_stdp = true;          // Enable spike-timing dependent plasticity
        agent_config.enable_neurogenesis = true;  // Enable neurogenesis
        agent_config.enable_pruning = true;       // Enable synaptic pruning
        
        learning_agent_ = std::make_unique<AutonomousLearningAgent>(agent_config);
        learning_agent_->initialize();
        
        is_initialized_ = true;
        std::cout << "✅ CentralController: Initialization complete!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "CentralController: Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void CentralController::initialize_neural_modules() {
    std::cout << "CentralController: Initializing neural modules..." << std::endl;
    
    // Create configurations
    NetworkConfig cognitive_config;
    cognitive_config.hidden_size = 512;
    cognitive_config.num_neurons = 2048;
    cognitive_config.hidden_hidden_prob = 0.05f;
    cognitive_config.input_hidden_prob = 0.15f;
    
    NetworkConfig motor_config;
    motor_config.hidden_size = 256;
    motor_config.num_neurons = 1024;
    motor_config.hidden_hidden_prob = 0.08f;
    motor_config.input_hidden_prob = 0.2f;
    
    // Create neural modules
    perception_module_ = std::make_shared<NeuralModule>("PerceptionNet", cognitive_config);
    planning_module_ = std::make_shared<NeuralModule>("PlanningNet", cognitive_config);
    motor_module_ = std::make_shared<NeuralModule>("MotorControlNet", motor_config);
    
    // Initialize the modules
    if (!perception_module_->initialize()) {
        throw std::runtime_error("Failed to initialize PerceptionNet");
    }
    if (!planning_module_->initialize()) {
        throw std::runtime_error("Failed to initialize PlanningNet");
    }
    if (!motor_module_->initialize()) {
        throw std::runtime_error("Failed to initialize MotorControlNet");
    }
    
    std::cout << "✅ Neural modules created and initialized" << std::endl;
}

void CentralController::initialize_task_modules() {
    std::cout << "CentralController: Initializing task modules..." << std::endl;
    
    // Create task-level modules
    cognitive_module_ = std::make_shared<CognitiveModule>(perception_module_, planning_module_);
    motor_task_module_ = std::make_shared<MotorModule>(motor_module_);
    
    // Initialize them
    cognitive_module_->initialize();
    motor_task_module_->initialize();
    
    std::cout << "✅ Task modules initialized" << std::endl;
}

// ============================================================================
// MAIN CONTROL INTERFACE
// ============================================================================

void CentralController::run(int cycles) {
    if (!is_initialized_) {
        std::cerr << "CentralController: Cannot run - not initialized!" << std::endl;
        return;
    }
    
    std::cout << "CentralController: Running " << cycles << " cognitive cycles..." << std::endl;
    
    for (int i = 0; i < cycles; ++i) {
        cycle_count_++;
        std::cout << "\n--- Cycle " << cycle_count_ << " ---" << std::endl;
        
        execute_cognitive_cycle();
        update_performance_metrics();
    }
    
    std::cout << "CentralController: Completed " << cycles << " cycles" << std::endl;
}

void CentralController::execute_cognitive_cycle() {
    const float dt = 0.1f; // 100ms per cycle
    
    // Update controller module
    neuro_controller_->update(dt);
    
    // **FIXED: Use autonomousLearningStep instead of update**
    // Update learning agent with autonomous learning step
    float learning_progress = learning_agent_->autonomousLearningStep(dt);
    std::cout << "Learning progress: " << learning_progress << std::endl;
    
    // Generate attention allocation
    std::unordered_map<std::string, float> attention_weights;
    attention_weights["PerceptionNet"] = 0.4f;
    attention_weights["PlanningNet"] = 0.4f;
    attention_weights["MotorControlNet"] = 0.2f;
    
    // Apply attention through individual module updates
    for (const auto& [module_name, weight] : attention_weights) {
        auto module = neuro_controller_->get_module(module_name);
        if (module) {
            // Apply attention weight through neuromodulator release
            neuro_controller_->release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, weight, module_name);
        }
    }
    
    // Coordinate neural activity
    neuro_controller_->coordinate_module_activities();
    
    // Update neural modules with time step
    std::vector<float> dummy_input(128, 0.1f); // Small input signal
    perception_module_->update(dt, dummy_input);
    planning_module_->update(dt, dummy_input);
    motor_module_->update(dt, dummy_input);
    
    std::cout << "CentralController: Cognitive cycle completed" << std::endl;
}

void CentralController::update_performance_metrics() {
    std::cout << "CentralController: Performance Metrics:" << std::endl;
    
    if (neuro_controller_) {
        std::cout << "  - Dopamine: " << neuro_controller_->get_concentration(NeuromodulatorType::DOPAMINE) << std::endl;
        std::cout << "  - Serotonin: " << neuro_controller_->get_concentration(NeuromodulatorType::SEROTONIN) << std::endl;
        std::cout << "  - Norepinephrine: " << neuro_controller_->get_concentration(NeuromodulatorType::NOREPINEPHRINE) << std::endl;
        std::cout << "  - System Coherence: " << neuro_controller_->get_system_coherence() << std::endl;
    }
    
    // Display module activity levels
    if (perception_module_) {
        auto perception_stats = perception_module_->get_stats();
        std::cout << "  - Perception Activity: " << perception_stats.mean_firing_rate << " Hz" << std::endl;
        std::cout << "  - Perception Active Ratio: " << perception_stats.neuron_activity_ratio << std::endl;
    }
    
    if (planning_module_) {
        auto planning_stats = planning_module_->get_stats();
        std::cout << "  - Planning Activity: " << planning_stats.mean_firing_rate << " Hz" << std::endl;
        std::cout << "  - Planning Active Ratio: " << planning_stats.neuron_activity_ratio << std::endl;
    }
    
    if (motor_module_) {
        auto motor_stats = motor_module_->get_stats();
        std::cout << "  - Motor Activity: " << motor_stats.mean_firing_rate << " Hz" << std::endl;
        std::cout << "  - Motor Active Ratio: " << motor_stats.neuron_activity_ratio << std::endl;
    }
}

// ============================================================================
// SYSTEM STATUS AND SHUTDOWN
// ============================================================================

void CentralController::shutdown() {
    if (!is_initialized_) return;
    
    std::cout << "CentralController: Shutting down..." << std::endl;
    
    if (learning_agent_) {
        learning_agent_->shutdown();
    }
    
    if (neuro_controller_) {
        neuro_controller_->emergency_stop();
    }
    
    // Reset modules
    perception_module_.reset();
    planning_module_.reset();
    motor_module_.reset();
    cognitive_module_.reset();
    motor_task_module_.reset();
    
    is_initialized_ = false;
    std::cout << "CentralController: Shutdown complete" << std::endl;
}

std::string CentralController::getSystemStatus() const {
    if (!is_initialized_) {
        return "System not initialized";
    }
    
    std::stringstream ss;
    ss << "Central Controller Status:\n";
    ss << "  Initialized: " << (is_initialized_ ? "Yes" : "No") << "\n";
    ss << "  Cycles completed: " << cycle_count_ << "\n";
    
    if (neuro_controller_) {
        ss << "  Controller active: Yes\n";
        ss << "  Registered modules: " << neuro_controller_->get_registered_modules().size() << "\n";
        ss << "  System coherence: " << neuro_controller_->get_system_coherence() << "\n";
    }
    
    if (learning_agent_) {
        ss << "  Learning agent active: Yes\n";
        ss << "  Learning enabled: " << (learning_agent_->isLearningActive() ? "Yes" : "No") << "\n";
        
        OperatingMode mode = learning_agent_->getCurrentMode();
        std::string mode_str = (mode == OperatingMode::AUTONOMOUS ? "AUTONOMOUS" : 
                               mode == OperatingMode::IDLE ? "IDLE" : "MANUAL");
        ss << "  Current mode: " << mode_str << "\n";
        
        std::string decision = learning_agent_->getCurrentDecision();
        if (!decision.empty()) {
            ss << "  Current action: " << decision << "\n";
        } else {
            ss << "  Current action: None\n";
        }
        
        ss << "  Decision confidence: " << std::fixed << std::setprecision(2) 
           << learning_agent_->getDecisionConfidence() << "\n";
    }
    
    // Module status
    ss << "  Neural Modules:\n";
    if (perception_module_) {
        ss << "    - PerceptionNet: Active\n";
    }
    if (planning_module_) {
        ss << "    - PlanningNet: Active\n";
    }
    if (motor_module_) {
        ss << "    - MotorControlNet: Active\n";
    }
    
    return ss.str();
}

float CentralController::getSystemPerformance() const {
    if (!is_initialized_ || !neuro_controller_) {
        return 0.0f;
    }
    
    // Calculate performance based on system coherence and activity
    float performance = neuro_controller_->get_system_coherence();
    
    // Factor in neural module activity
    float total_activity = 0.0f;
    int active_modules = 0;
    
    if (perception_module_) {
        auto stats = perception_module_->get_stats();
        total_activity += stats.mean_firing_rate;
        active_modules++;
    }
    
    if (planning_module_) {
        auto stats = planning_module_->get_stats();
        total_activity += stats.mean_firing_rate;
        active_modules++;
    }
    
    if (motor_module_) {
        auto stats = motor_module_->get_stats();
        total_activity += stats.mean_firing_rate;
        active_modules++;
    }
    
    if (active_modules > 0) {
        float average_activity = total_activity / active_modules;
        performance = (performance + average_activity) / 2.0f;
    }
    
    return std::min(1.0f, performance);
}