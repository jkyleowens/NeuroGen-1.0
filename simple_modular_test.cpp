// ============================================================================
// SIMPLE MODULAR COMPONENT TEST
// Test basic functionality of compiled modular components
// ============================================================================

#include <iostream>
#include <memory>
#include <vector>

// Core NeuroGen components (minimal test)
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/NetworkConfig.h"

int main() {
    std::cout << "🔬 NeuroGen 0.5.5 - Simple Modular Component Test\n";
    std::cout << "================================================\n" << std::endl;
    
    try {
        // Test 1: Create NetworkConfig
        std::cout << "Test 1: Creating NetworkConfig..." << std::endl;
        NetworkConfig config;
        config.num_neurons = 16;
        config.enable_neurogenesis = false;
        config.enable_stdp = false;
        std::cout << "✅ NetworkConfig created successfully" << std::endl;
        
        // Test 2: Create ControllerModule
        std::cout << "\nTest 2: Creating ControllerModule..." << std::endl;
        ControllerConfig controller_config;
        auto controller = std::make_unique<ControllerModule>(controller_config);
        std::cout << "✅ ControllerModule created successfully" << std::endl;
        
        // Test 3: Create EnhancedNeuralModule
        std::cout << "\nTest 3: Creating EnhancedNeuralModule..." << std::endl;
        auto neural_module = std::make_unique<EnhancedNeuralModule>("test_module", config);
        std::cout << "✅ EnhancedNeuralModule created successfully" << std::endl;
        
        // Test 4: Basic operations
        std::cout << "\nTest 4: Testing basic operations..." << std::endl;
        
        // Test neural module processing
        std::vector<float> test_input = {0.1f, 0.2f, 0.3f};
        neural_module->update(0.1f, test_input, 0.0f);
        std::cout << "✅ Neural module update completed" << std::endl;
        
        // Test controller update
        controller->update(0.1f);
        std::cout << "✅ Controller update completed" << std::endl;
        
        // Test 5: Performance metrics
        std::cout << "\nTest 5: Testing performance metrics..." << std::endl;
        auto metrics = neural_module->getPerformanceMetrics();
        std::cout << "✅ Retrieved " << metrics.size() << " performance metrics" << std::endl;
        
        // Test 6: Module state
        std::cout << "\nTest 6: Testing module state..." << std::endl;
        bool is_active = neural_module->is_active();
        std::cout << "✅ Module active state: " << (is_active ? "Active" : "Inactive") << std::endl;
        
        std::cout << "\n🎉 SUCCESS: All modular component tests passed!" << std::endl;
        std::cout << "🔗 Modular neural network architecture is functional" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test Failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred during testing" << std::endl;
        return 1;
    }
}
