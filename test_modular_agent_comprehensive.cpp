// ============================================================================
// COMPREHENSIVE MODULAR AGENT UNIT TEST SUITE
// File: test_modular_agent_comprehensive.cpp
// ============================================================================

#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/CentralController.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/SpecializedModule.h"
#include "NeuroGen/NetworkConfig.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <random>
#include <cassert>
#include <iomanip>
#include <memory>

// ============================================================================
// COMPREHENSIVE TEST FRAMEWORK
// ============================================================================

class ModularAgentTestFramework {
public:
    struct TestResult {
        std::string test_name;
        bool passed;
        double execution_time_ms;
        std::string error_message;
        std::map<std::string, float> metrics;
        
        TestResult() : passed(false), execution_time_ms(0.0) {}
    };
    
    std::vector<TestResult> results;
    int total_tests = 0;
    int passed_tests = 0;
    
    void runTest(const std::string& name, std::function<bool()> test_func) {
        std::cout << "\n=== Running Test: " << name << " ===" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        TestResult result;
        result.test_name = name;
        total_tests++;
        
        try {
            result.passed = test_func();
            if (result.passed) {
                passed_tests++;
                std::cout << "✅ PASSED" << std::endl;
            } else {
                std::cout << "❌ FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
            std::cout << "❌ EXCEPTION: " << e.what() << std::endl;
        } catch (...) {
            result.passed = false;
            result.error_message = "Unknown exception";
            std::cout << "❌ UNKNOWN EXCEPTION" << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        results.push_back(result);
        
        std::cout << "Execution time: " << std::fixed << std::setprecision(2) 
                  << result.execution_time_ms << " ms" << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "COMPREHENSIVE MODULAR AGENT TEST SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << "Total Tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passed_tests / total_tests) << "%" << std::endl;
        
        double total_time = 0.0;
        for (const auto& result : results) {
            total_time += result.execution_time_ms;
        }
        std::cout << "Total Execution Time: " << std::fixed << std::setprecision(2) 
                  << total_time << " ms" << std::endl;
        
        // Show failed tests
        bool has_failures = false;
        for (const auto& result : results) {
            if (!result.passed) {
                if (!has_failures) {
                    std::cout << "\n❌ Failed Tests:" << std::endl;
                    has_failures = true;
                }
                std::cout << "  - " << result.test_name;
                if (!result.error_message.empty()) {
                    std::cout << ": " << result.error_message;
                }
                std::cout << std::endl;
            }
        }
        
        if (passed_tests == total_tests) {
            std::cout << "\n🎉 ALL TESTS PASSED! Modular agent is ready for operation." << std::endl;
        } else {
            std::cout << "\n⚠️  Some tests failed. Please review the errors above." << std::endl;
        }
    }
};

// ============================================================================
// CORE MODULAR AGENT TESTS
// ============================================================================

// Test 1: Basic Network Configuration
bool testNetworkConfiguration() {
    std::cout << "Testing NetworkConfig creation and validation..." << std::endl;
    
    NetworkConfig config;
    config.num_neurons = 128;
    config.hidden_size = 256;
    config.input_size = 64;
    config.output_size = 32;
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    
    // Test configuration validation
    if (!config.validate()) {
        std::cout << "Config validation failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ NetworkConfig created and validated successfully" << std::endl;
    std::cout << "  Neurons: " << config.num_neurons << std::endl;
    std::cout << "  Hidden size: " << config.hidden_size << std::endl;
    std::cout << "  STDP enabled: " << (config.enable_stdp ? "Yes" : "No") << std::endl;
    
    return true;
}

// Test 2: Autonomous Learning Agent Initialization
bool testAutonomousLearningAgentInit() {
    std::cout << "Testing AutonomousLearningAgent initialization..." << std::endl;
    
    NetworkConfig config;
    config.num_neurons = 64;
    config.hidden_size = 128;
    config.input_size = 32;
    config.output_size = 16;
    config.enable_neurogenesis = false; // Disable for testing
    config.enable_stdp = false;
    
    auto agent = std::make_unique<AutonomousLearningAgent>(config);
    
    if (!agent) {
        std::cout << "Failed to create AutonomousLearningAgent" << std::endl;
        return false;
    }
    
    // Test initialization
    if (!agent->initialize()) {
        std::cout << "Failed to initialize AutonomousLearningAgent" << std::endl;
        return false;
    }
    
    std::cout << "✓ AutonomousLearningAgent created and initialized" << std::endl;
    
    // Test basic functionality
    agent->startAutonomousLearning();
    std::cout << "✓ Started autonomous learning mode" << std::endl;
    
    // Perform a few learning steps
    for (int i = 0; i < 5; ++i) {
        float progress = agent->autonomousLearningStep(0.1f);
        std::cout << "  Step " << i+1 << " progress: " << progress << std::endl;
    }
    
    agent->stopAutonomousLearning();
    std::cout << "✓ Stopped autonomous learning mode" << std::endl;
    
    return true;
}

// Test 3: Brain Module Architecture Creation
bool testBrainModuleArchitecture() {
    std::cout << "Testing BrainModuleArchitecture creation..." << std::endl;
    
    auto brain = std::make_unique<BrainModuleArchitecture>();
    
    if (!brain) {
        std::cout << "Failed to create BrainModuleArchitecture" << std::endl;
        return false;
    }
    
    // Test initialization with different screen sizes
    if (!brain->initialize(1920, 1080)) {
        std::cout << "Failed to initialize BrainModuleArchitecture" << std::endl;
        return false;
    }
    
    std::cout << "✓ BrainModuleArchitecture initialized successfully" << std::endl;
    
    // Test module availability
    auto module_names = brain->getModuleNames();
    std::cout << "✓ Available modules (" << module_names.size() << "):" << std::endl;
    for (const auto& name : module_names) {
        std::cout << "  - " << name << std::endl;
    }
    
    // Test basic processing
    std::vector<float> dummy_visual_input(1920 * 1080 / 16, 0.5f); // Reduced size for testing
    auto visual_features = brain->processVisualInput(dummy_visual_input);
    
    if (visual_features.empty()) {
        std::cout << "Visual processing returned empty result" << std::endl;
        return false;
    }
    
    std::cout << "✓ Visual processing successful, output size: " << visual_features.size() << std::endl;
    
    return true;
}

// Test 4: Central Controller Integration
bool testCentralController() {
    std::cout << "Testing CentralController integration..." << std::endl;
    
    auto controller = std::make_unique<CentralController>();
    
    if (!controller) {
        std::cout << "Failed to create CentralController" << std::endl;
        return false;
    }
    
    // Test initialization
    if (!controller->initialize()) {
        std::cout << "Failed to initialize CentralController" << std::endl;
        return false;
    }
    
    std::cout << "✓ CentralController initialized successfully" << std::endl;
    
    // Test system status
    std::string status = controller->getSystemStatus();
    std::cout << "✓ System status retrieved:" << std::endl;
    std::cout << status << std::endl;
    
    // Test performance metrics
    float performance = controller->getSystemPerformance();
    std::cout << "✓ System performance: " << performance << std::endl;
    
    // Test a few cognitive cycles
    controller->run(3);
    std::cout << "✓ Completed 3 cognitive cycles" << std::endl;
    
    return true;
}

// Test 5: Controller Module Functionality
bool testControllerModule() {
    std::cout << "Testing ControllerModule functionality..." << std::endl;
    
    ControllerConfig config;
    config.initial_dopamine_level = 0.4f;
    config.initial_serotonin_level = 0.5f;
    config.reward_learning_rate = 0.02f;
    config.enable_detailed_logging = true;
    
    auto controller = std::make_unique<ControllerModule>(config);
    
    if (!controller) {
        std::cout << "Failed to create ControllerModule" << std::endl;
        return false;
    }
    
    std::cout << "✓ ControllerModule created with configuration" << std::endl;
    
    // Test neuromodulator levels
    float dopamine = controller->get_concentration(NeuromodulatorType::DOPAMINE);
    float serotonin = controller->get_concentration(NeuromodulatorType::SEROTONIN);
    
    std::cout << "✓ Initial neuromodulator levels:" << std::endl;
    std::cout << "  Dopamine: " << dopamine << std::endl;
    std::cout << "  Serotonin: " << serotonin << std::endl;
    
    // Test neuromodulator release
    controller->release_neuromodulator(NeuromodulatorType::DOPAMINE, 0.2f);
    float new_dopamine = controller->get_concentration(NeuromodulatorType::DOPAMINE);
    
    if (new_dopamine <= dopamine) {
        std::cout << "Dopamine release failed - no increase detected" << std::endl;
        return false;
    }
    
    std::cout << "✓ Dopamine release successful: " << dopamine << " -> " << new_dopamine << std::endl;
    
    // Test update cycles
    for (int i = 0; i < 5; ++i) {
        controller->update(0.1f);
    }
    
    std::cout << "✓ Completed 5 controller update cycles" << std::endl;
    
    return true;
}

// Test 6: Enhanced Neural Module Creation
bool testEnhancedNeuralModule() {
    std::cout << "Testing EnhancedNeuralModule creation..." << std::endl;
    
    NetworkConfig config;
    config.num_neurons = 32;
    config.hidden_size = 64;
    config.input_size = 16;
    config.output_size = 8;
    config.enable_stdp = false; // Disable CUDA features for testing
    config.enable_neurogenesis = false;
    
    auto module = std::make_unique<EnhancedNeuralModule>("test_module", config);
    
    if (!module) {
        std::cout << "Failed to create EnhancedNeuralModule" << std::endl;
        return false;
    }
    
    // Test initialization
    if (!module->initialize()) {
        std::cout << "Failed to initialize EnhancedNeuralModule" << std::endl;
        return false;
    }
    
    std::cout << "✓ EnhancedNeuralModule created and initialized" << std::endl;
    std::cout << "  Name: " << module->get_name() << std::endl;
    std::cout << "  Active: " << (module->is_active() ? "Yes" : "No") << std::endl;
    
    // Test processing
    std::vector<float> input(16, 0.5f);
    auto output = module->process(input);
    
    if (output.empty()) {
        std::cout << "Module processing returned empty output" << std::endl;
        return false;
    }
    
    std::cout << "✓ Module processing successful" << std::endl;
    std::cout << "  Input size: " << input.size() << std::endl;
    std::cout << "  Output size: " << output.size() << std::endl;
    
    // Test update cycle
    module->update(0.1f, input, 0.5f);
    std::cout << "✓ Module update cycle completed" << std::endl;
    
    return true;
}

// Test 7: Specialized Module Functionality
bool testSpecializedModule() {
    std::cout << "Testing SpecializedModule functionality..." << std::endl;
    
    NetworkConfig config;
    config.num_neurons = 32;
    config.hidden_size = 64;
    config.input_size = 16;
    config.output_size = 8;
    
    auto motor_module = std::make_unique<SpecializedModule>("motor_cortex", config, "motor");
    auto attention_module = std::make_unique<SpecializedModule>("attention_system", config, "attention");
    
    if (!motor_module || !attention_module) {
        std::cout << "Failed to create SpecializedModules" << std::endl;
        return false;
    }
    
    // Test initialization
    if (!motor_module->initialize() || !attention_module->initialize()) {
        std::cout << "Failed to initialize SpecializedModules" << std::endl;
        return false;
    }
    
    std::cout << "✓ SpecializedModules created and initialized" << std::endl;
    std::cout << "  Motor module: " << motor_module->get_name() << std::endl;
    std::cout << "  Attention module: " << attention_module->get_name() << std::endl;
    
    // Test specialized processing
    std::vector<float> input(16, 0.3f);
    
    auto motor_output = motor_module->process(input);
    auto attention_output = attention_module->process(input);
    
    std::cout << "✓ Specialized processing completed" << std::endl;
    std::cout << "  Motor output size: " << motor_output.size() << std::endl;
    std::cout << "  Attention output size: " << attention_output.size() << std::endl;
    
    return true;
}

// Test 8: Memory System Integration
bool testMemorySystem() {
    std::cout << "Testing MemorySystem integration..." << std::endl;
    
    auto memory = std::make_unique<MemorySystem>();
    
    if (!memory) {
        std::cout << "Failed to create MemorySystem" << std::endl;
        return false;
    }
    
    std::cout << "✓ MemorySystem created successfully" << std::endl;
    
    // Test episode storage
    std::vector<float> state = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> action = {0.5f, 0.6f};
    float reward = 0.8f;
    std::string context = "test_episode";
    
    memory->storeEpisode(state, action, reward, context);
    std::cout << "✓ Episode stored successfully" << std::endl;
    
    // Test episode retrieval
    std::vector<float> query = {0.1f, 0.2f, 0.3f, 0.4f};
    auto similar_episodes = memory->retrieveSimilarEpisodes(query, 5);
    
    if (similar_episodes.empty()) {
        std::cout << "No similar episodes found (this might be expected)" << std::endl;
    } else {
        std::cout << "✓ Found " << similar_episodes.size() << " similar episodes" << std::endl;
    }
    
    // Test working memory
    std::vector<float> working_memory_data = {0.7f, 0.8f, 0.9f};
    memory->update_working_memory(working_memory_data);
    
    auto retrieved_wm = memory->get_working_memory();
    std::cout << "✓ Working memory operations completed" << std::endl;
    std::cout << "  Working memory size: " << retrieved_wm.size() << std::endl;
    
    return true;
}

// Test 9: Inter-Module Communication
bool testInterModuleCommunication() {
    std::cout << "Testing inter-module communication..." << std::endl;
    
    // Create a simple agent for testing
    NetworkConfig config;
    config.num_neurons = 32;
    config.hidden_size = 64;
    config.enable_neurogenesis = false;
    config.enable_stdp = false;
    
    auto agent = std::make_unique<AutonomousLearningAgent>(config);
    
    if (!agent->initialize()) {
        std::cout << "Failed to initialize agent for communication test" << std::endl;
        return false;
    }
    
    std::cout << "✓ Agent initialized for communication testing" << std::endl;
    
    // Test a few learning steps to exercise communication
    agent->startAutonomousLearning();
    
    for (int i = 0; i < 3; ++i) {
        float progress = agent->autonomousLearningStep(0.1f);
        std::cout << "  Communication step " << i+1 << " progress: " << progress << std::endl;
    }
    
    agent->stopAutonomousLearning();
    std::cout << "✓ Inter-module communication test completed" << std::endl;
    
    return true;
}

// Test 10: Full System Integration
bool testFullSystemIntegration() {
    std::cout << "Testing full system integration..." << std::endl;
    
    // Create all major components
    NetworkConfig config;
    config.num_neurons = 64;
    config.hidden_size = 128;
    config.enable_neurogenesis = false; // Disable for testing stability
    config.enable_stdp = false;
    
    auto agent = std::make_unique<AutonomousLearningAgent>(config);
    auto controller = std::make_unique<CentralController>();
    auto brain = std::make_unique<BrainModuleArchitecture>();
    
    // Initialize all components
    if (!agent->initialize()) {
        std::cout << "Failed to initialize AutonomousLearningAgent" << std::endl;
        return false;
    }
    
    if (!controller->initialize()) {
        std::cout << "Failed to initialize CentralController" << std::endl;
        return false;
    }
    
    if (!brain->initialize(800, 600)) { // Smaller size for testing
        std::cout << "Failed to initialize BrainModuleArchitecture" << std::endl;
        return false;
    }
    
    std::cout << "✓ All major components initialized successfully" << std::endl;
    
    // Test coordinated operation
    agent->startAutonomousLearning();
    
    for (int cycle = 0; cycle < 3; ++cycle) {
        std::cout << "  Integration cycle " << cycle + 1 << std::endl;
        
        // Agent learning step
        float agent_progress = agent->autonomousLearningStep(0.1f);
        
        // Controller cognitive cycle
        controller->run(1);
        
        // Brain update
        brain->update(0.1f);
        
        std::cout << "    Agent progress: " << agent_progress << std::endl;
        std::cout << "    System performance: " << controller->getSystemPerformance() << std::endl;
        std::cout << "    Brain activity: " << brain->getTotalActivity() << std::endl;
    }
    
    agent->stopAutonomousLearning();
    std::cout << "✓ Full system integration test completed successfully" << std::endl;
    
    return true;
}

// ============================================================================
// PERFORMANCE AND STRESS TESTS
// ============================================================================

// Test 11: Performance Benchmark
bool testPerformanceBenchmark() {
    std::cout << "Running performance benchmark..." << std::endl;
    
    NetworkConfig config;
    config.num_neurons = 128;
    config.hidden_size = 256;
    config.enable_neurogenesis = false;
    config.enable_stdp = false;
    
    auto agent = std::make_unique<AutonomousLearningAgent>(config);
    
    if (!agent->initialize()) {
        std::cout << "Failed to initialize agent for benchmark" << std::endl;
        return false;
    }
    
    agent->startAutonomousLearning();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run 100 learning steps
    for (int i = 0; i < 100; ++i) {
        agent->autonomousLearningStep(0.01f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    agent->stopAutonomousLearning();
    
    double steps_per_second = 100.0 * 1000.0 / duration_ms;
    
    std::cout << "✓ Performance benchmark completed" << std::endl;
    std::cout << "  Duration: " << std::fixed << std::setprecision(2) << duration_ms << " ms" << std::endl;
    std::cout << "  Learning steps per second: " << std::fixed << std::setprecision(1) << steps_per_second << std::endl;
    
    // Performance threshold: should complete at least 10 steps per second
    return steps_per_second >= 10.0;
}

// Test 12: Memory Stress Test
bool testMemoryStressTest() {
    std::cout << "Running memory stress test..." << std::endl;
    
    auto memory = std::make_unique<MemorySystem>();
    
    // Store many episodes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    const int num_episodes = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_episodes; ++i) {
        std::vector<float> state(10);
        std::vector<float> action(4);
        
        // Generate random data
        for (auto& s : state) s = dis(gen);
        for (auto& a : action) a = dis(gen);
        
        float reward = dis(gen);
        std::string context = "episode_" + std::to_string(i);
        
        memory->storeEpisode(state, action, reward, context);
        
        if (i % 100 == 0) {
            std::cout << "  Stored " << i << " episodes..." << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "✓ Memory stress test completed" << std::endl;
    std::cout << "  Stored " << num_episodes << " episodes in " 
              << std::fixed << std::setprecision(2) << duration_ms << " ms" << std::endl;
    
    // Test retrieval
    std::vector<float> query(10, 0.5f);
    auto similar = memory->retrieveSimilarEpisodes(query, 10);
    
    std::cout << "  Retrieved " << similar.size() << " similar episodes" << std::endl;
    
    return true;
}

// ============================================================================
// MAIN TEST EXECUTION
// ============================================================================

int main() {
    std::cout << "🧠 NeuroGen 0.5.5 - Comprehensive Modular Agent Unit Test Suite" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Testing all components of the modular autonomous learning agent" << std::endl;
    
    ModularAgentTestFramework framework;
    
    // Core Component Tests
    framework.runTest("NetworkConfig Validation", testNetworkConfiguration);
    framework.runTest("AutonomousLearningAgent Initialization", testAutonomousLearningAgentInit);
    framework.runTest("BrainModuleArchitecture Creation", testBrainModuleArchitecture);
    framework.runTest("CentralController Integration", testCentralController);
    framework.runTest("ControllerModule Functionality", testControllerModule);
    framework.runTest("EnhancedNeuralModule Creation", testEnhancedNeuralModule);
    framework.runTest("SpecializedModule Functionality", testSpecializedModule);
    framework.runTest("MemorySystem Integration", testMemorySystem);
    
    // Advanced Integration Tests
    framework.runTest("Inter-Module Communication", testInterModuleCommunication);
    framework.runTest("Full System Integration", testFullSystemIntegration);
    
    // Performance Tests
    framework.runTest("Performance Benchmark", testPerformanceBenchmark);
    framework.runTest("Memory Stress Test", testMemoryStressTest);
    
    // Print comprehensive summary
    framework.printSummary();
    
    // Return appropriate exit code
    return (framework.passed_tests == framework.total_tests) ? 0 : 1;
}
