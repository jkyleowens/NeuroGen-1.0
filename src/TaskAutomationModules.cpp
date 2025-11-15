#include "NeuroGen/TaskAutomationModules.h"
#include <iostream>

// Note: The TaskModule base class defined in the header is abstract
// and does not require a separate implementation file since it has no
// non-pure virtual functions to define.

// --- Implementation for CognitiveModule ---

// The constructor was already defined in the header, so no separate
// implementation is needed here unless it becomes more complex.
// If you were to define it here, it would look like this:
/*
CognitiveModule::CognitiveModule(std::shared_ptr<NeuralModule> perception, std::shared_ptr<NeuralModule> planning)
    : perception_module_(perception), planning_module_(planning) {
    // Constructor body
}
*/

// Implementation of the initialize method for CognitiveModule
void CognitiveModule::initialize() {
    std::cout << "Initializing Cognitive Module..." << std::endl;
    if (perception_module_ && planning_module_) {
        std::cout << "  - Perception Module: " << perception_module_->get_name() << " is linked." << std::endl;
        std::cout << "  - Planning Module: " << planning_module_->get_name() << " is linked." << std::endl;
        // In a real scenario, you would establish connections here.
        // For example: network->connect(perception_module_->get_name(), "output", planning_module_->get_name(), "input");
    } else {
        std::cerr << "  - Warning: One or both neural modules are null in CognitiveModule." << std::endl;
    }
}


// --- Implementation for MotorModule ---

// Similar to CognitiveModule, the constructor is simple enough
// to be fully defined in the header.
/*
MotorModule::MotorModule(std::shared_ptr<NeuralModule> motor_control)
    : motor_control_module_(motor_control) {
    // Constructor body
}
*/

// Implementation of the initialize method for MotorModule
void MotorModule::initialize() {
    std::cout << "Initializing Motor Module..." << std::endl;
    if (motor_control_module_) {
        std::cout << "  - Motor Control Module: " << motor_control_module_->get_name() << " is configured." << std::endl;
        // Logic to configure motor control outputs, e.g., mapping neurons to actuators.
    } else {
        std::cerr << "  - Warning: Motor control neural module is null in MotorModule." << std::endl;
    }
}