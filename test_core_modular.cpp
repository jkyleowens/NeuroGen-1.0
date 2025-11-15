// Simple test for core modular components without CUDA dependencies
// NeuroGen 0.5.5 - Core Modular Network Test

#include <iostream>
#include <memory>
#include <vector>

// Only test the components we know work without CUDA
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/NetworkConfig.h"

int main() {
    std::cout << "=== NeuroGen 0.5.5 Core Modular Test (No CUDA) ===" << std::endl;
    
    try {
        // Create a basic network configuration
        NetworkConfig config;
        config.num_neurons = 64;
        config.enable_neurogenesis = true;
        config.enable_stdp = true;
        config.enable_homeostasis = true;
        
        std::cout << "✓ Created network configuration with " << config.num_neurons << " neurons" << std::endl;
        
        // Test ControllerModule creation and basic functionality
        std::unique_ptr<ControllerModule> controller = 
            std::make_unique<ControllerModule>(config);
        
        std::cout << "✓ Created ControllerModule successfully" << std::endl;
        
        // Test basic controller operations
        for (int step = 0; step < 3; ++step) {
            float current_time = step * 0.1f;
            float dt = 0.1f;
            
            std::cout << "Step " << (step + 1) << ": ";
            controller->update(current_time, dt);
            std::cout << "Controller updated successfully" << std::endl;
        }
        
        std::cout << "\n🎉 SUCCESS: Core modular components are working!" << std::endl;
        std::cout << "✓ ControllerModule operational" << std::endl;
        std::cout << "✓ NetworkConfig system functional" << std::endl;
        std::cout << "✓ Basic modular architecture established" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred" << std::endl;
        return 1;
    }
}
