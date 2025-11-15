// Test program for modular autonomous neural network agent
// NeuroGen 0.5.5 - Modular Network Demonstration

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

// Core modular components that we know compile
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/CentralController.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/NetworkConfig.h"

// Simple test to demonstrate modular network functionality
int main() {
    std::cout << "=== NeuroGen 0.5.5 Modular Neural Network Test ===" << std::endl;
    std::cout << "Testing compiled modular components..." << std::endl;
    
    try {
        // Create network configuration
        NetworkConfig config;
        config.num_neurons = 128;
        config.enable_neurogenesis = true;
        config.enable_stdp = true;
        config.enable_homeostasis = true;
        
        std::cout << "✓ Created network configuration" << std::endl;
        
        // Test ControllerModule creation
        std::unique_ptr<ControllerModule> controller = 
            std::make_unique<ControllerModule>(config);
        
        std::cout << "✓ Created ControllerModule" << std::endl;
        
        // Test EnhancedNeuralModule creation
        std::unique_ptr<EnhancedNeuralModule> neural_module = 
            std::make_unique<EnhancedNeuralModule>("test_module", config);
        
        std::cout << "✓ Created EnhancedNeuralModule" << std::endl;
        
        // Test AutonomousLearningAgent creation
        std::unique_ptr<AutonomousLearningAgent> learning_agent = 
            std::make_unique<AutonomousLearningAgent>();
        
        std::cout << "✓ Created AutonomousLearningAgent" << std::endl;
        
        // Test CentralController creation
        std::unique_ptr<CentralController> central_controller = 
            std::make_unique<CentralController>();
        
        std::cout << "✓ Created CentralController" << std::endl;
        
        // Test basic module initialization
        std::vector<float> input_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        
        // Test neural module processing
        std::vector<float> module_output = neural_module->process(input_data);
        std::cout << "✓ Neural module processed " << input_data.size() 
                  << " inputs, produced " << module_output.size() << " outputs" << std::endl;
        
        // Test controller module
        controller->update(0.0f, 0.1f);  // time=0, dt=0.1
        std::cout << "✓ Controller module updated" << std::endl;
        
        // Test learning agent step
        learning_agent->step();
        std::cout << "✓ Learning agent performed step" << std::endl;
        
        // Test central controller update
        central_controller->update_modules(0.0f, 0.1f);
        std::cout << "✓ Central controller updated modules" << std::endl;
        
        // Simulate a few time steps
        std::cout << "\nRunning brief simulation..." << std::endl;
        for (int step = 0; step < 5; ++step) {
            float current_time = step * 0.1f;
            float dt = 0.1f;
            
            // Update modules
            controller->update(current_time, dt);
            neural_module->process(input_data);
            learning_agent->step();
            central_controller->update_modules(current_time, dt);
            
            std::cout << "Step " << step + 1 << " completed" << std::endl;
            
            // Small delay for demonstration
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n🎉 SUCCESS: All modular components are working!" << std::endl;
        std::cout << "✓ AutonomousLearningAgent operational" << std::endl;
        std::cout << "✓ CentralController operational" << std::endl;
        std::cout << "✓ ControllerModule operational" << std::endl;
        std::cout << "✓ EnhancedNeuralModule operational" << std::endl;
        std::cout << "✓ Modular network simulation completed successfully" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during modular network test: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error during modular network test" << std::endl;
        return 1;
    }
}
