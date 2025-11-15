// Minimal test to verify compilation and basic functionality
#include <iostream>
#include <memory>
#include <vector>

// Include basic headers
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/BrainModuleArchitecture.h"

int main() {
    std::cout << "=== Minimal Modular Agent Test ===" << std::endl;
    
    try {
        // Test 1: NetworkConfig creation
        std::cout << "Test 1: Creating NetworkConfig..." << std::endl;
        NetworkConfig config;
        config.input_size = 10;
        config.hidden_size = 50;
        config.output_size = 5;
        config.learning_rate = 0.01f;
        std::cout << "✓ NetworkConfig created successfully" << std::endl;
        
        // Test 2: AutonomousLearningAgent creation
        std::cout << "Test 2: Creating AutonomousLearningAgent..." << std::endl;
        auto agent = std::make_unique<AutonomousLearningAgent>(config);
        std::cout << "✓ AutonomousLearningAgent created successfully" << std::endl;
        
        // Test 3: BrainModuleArchitecture creation
        std::cout << "Test 3: Creating BrainModuleArchitecture..." << std::endl;
        auto brain = std::make_unique<BrainModuleArchitecture>(config);
        std::cout << "✓ BrainModuleArchitecture created successfully" << std::endl;
        
        // Test 4: Simple learning step
        std::cout << "Test 4: Performing learning step..." << std::endl;
        std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
        std::vector<float> output = agent->step(input, 0.1f); // Small reward
        
        std::cout << "Input size: " << input.size() << ", Output size: " << output.size() << std::endl;
        std::cout << "✓ Learning step completed successfully" << std::endl;
        
        std::cout << "\n=== All Tests Passed Successfully! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
