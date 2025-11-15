// ============================================================================
// BRAIN MODULE ARCHITECTURE COMPREHENSIVE TEST SUITE
// File: test_brain_module_architecture.cpp
// ============================================================================

#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkConfig.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <random>
#include <cassert>
#include <iomanip>

// ============================================================================
// TEST FRAMEWORK
// ============================================================================

class TestFramework {
public:
    struct TestResult {
        std::string test_name;
        bool passed;
        double execution_time_ms;
        std::string error_message;
        std::map<std::string, float> metrics;
    };
    
    std::vector<TestResult> results;
    
    void runTest(const std::string& name, std::function<bool()> test_func) {
        std::cout << "\n=== Running Test: " << name << " ===" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        TestResult result;
        result.test_name = name;
        
        try {
            result.passed = test_func();
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        results.push_back(result);
        
        std::cout << "Result: " << (result.passed ? "PASSED" : "FAILED") 
                  << " (" << std::fixed << std::setprecision(2) 
                  << result.execution_time_ms << "ms)" << std::endl;
        
        if (!result.passed && !result.error_message.empty()) {
            std::cout << "Error: " << result.error_message << std::endl;
        }
    }
    
    void printSummary() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST SUITE SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        int passed = 0, failed = 0;
        double total_time = 0.0;
        
        for (const auto& result : results) {
            if (result.passed) passed++;
            else failed++;
            total_time += result.execution_time_ms;
            
            std::cout << std::setw(40) << std::left << result.test_name 
                      << " | " << (result.passed ? "PASS" : "FAIL")
                      << " | " << std::setw(8) << std::right 
                      << std::fixed << std::setprecision(2) 
                      << result.execution_time_ms << "ms" << std::endl;
        }
        
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "Total Tests: " << (passed + failed) 
                  << " | Passed: " << passed 
                  << " | Failed: " << failed 
                  << " | Total Time: " << total_time << "ms" << std::endl;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passed / (passed + failed)) << "%" << std::endl;
    }
};

// ============================================================================
// TEST UTILITIES
// ============================================================================

std::vector<float> generateRandomInput(size_t size, float min_val = 0.0f, float max_val = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    std::vector<float> input(size);
    for (auto& val : input) {
        val = dis(gen);
    }
    return input;
}

std::vector<float> generateVisualInput(int width, int height) {
    return generateRandomInput(width * height / 16); // Downsampled
}

bool validateOutput(const std::vector<float>& output, size_t expected_size, 
                   float min_val = -10.0f, float max_val = 10.0f) {
    if (output.size() != expected_size) {
        std::cout << "Size mismatch: expected " << expected_size 
                  << ", got " << output.size() << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            std::cout << "Invalid value at index " << i << ": " << output[i] << std::endl;
            return false;
        }
        if (output[i] < min_val || output[i] > max_val) {
            std::cout << "Value out of range at index " << i << ": " << output[i] 
                      << " (expected [" << min_val << ", " << max_val << "])" << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// BRAIN MODULE ARCHITECTURE TESTS
// ============================================================================

bool testBrainArchitectureInitialization() {
    BrainModuleArchitecture brain;
    
    // Test initialization
    bool init_success = brain.initialize(1920, 1080);
    if (!init_success) {
        std::cout << "Failed to initialize brain architecture" << std::endl;
        return false;
    }
    
    // Test module creation
    auto module_names = brain.getModuleNames();
    std::vector<std::string> expected_modules = {
        "visual_cortex", "comprehension_module", "executive_function",
        "memory_module", "central_controller", "output_module",
        "motor_cortex", "reward_system", "attention_system"
    };
    
    if (module_names.size() != expected_modules.size()) {
        std::cout << "Expected " << expected_modules.size() << " modules, got " 
                  << module_names.size() << std::endl;
        return false;
    }
    
    // Verify all expected modules exist
    for (const auto& expected : expected_modules) {
        bool found = false;
        for (const auto& actual : module_names) {
            if (actual == expected) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cout << "Missing expected module: " << expected << std::endl;
            return false;
        }
    }
    
    // Test module access
    for (const auto& name : module_names) {
        auto module = brain.getModule(name);
        if (!module) {
            std::cout << "Failed to get module: " << name << std::endl;
            return false;
        }
    }
    
    std::cout << "Successfully initialized " << module_names.size() << " modules" << std::endl;
    return true;
}

bool testVisualProcessingPipeline() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Generate test visual input
    auto visual_input = generateVisualInput(1920, 1080);
    
    // Process visual input
    auto visual_features = brain.processVisualInput(visual_input);
    
    // Validate output
    if (!validateOutput(visual_features, 1800, -5.0f, 5.0f)) { // Expected visual feature size
        return false;
    }
    
    std::cout << "Visual processing: " << visual_input.size() 
              << " -> " << visual_features.size() << " features" << std::endl;
    
    // Test multiple processing steps
    for (int i = 0; i < 5; ++i) {
        auto new_input = generateVisualInput(1920, 1080);
        auto new_features = brain.processVisualInput(new_input);
        
        if (!validateOutput(new_features, visual_features.size(), -5.0f, 5.0f)) {
            std::cout << "Failed on iteration " << i << std::endl;
            return false;
        }
    }
    
    return true;
}

bool testDecisionMakingPipeline() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Generate test inputs
    auto context_input = generateRandomInput(1800); // Visual feature size
    auto goals = generateRandomInput(64); // Goal size
    
    // Execute decision making
    auto decisions = brain.executeDecisionMaking(context_input, goals);
    
    // Validate output
    if (!validateOutput(decisions, 256, -5.0f, 5.0f)) { // Expected decision size
        return false;
    }
    
    std::cout << "Decision making: context(" << context_input.size() 
              << ") + goals(" << goals.size() 
              << ") -> " << decisions.size() << " decisions" << std::endl;
    
    return true;
}

bool testMotorOutputGeneration() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Generate test decision input
    auto decision_input = generateRandomInput(256); // Decision size
    
    // Generate motor output
    auto motor_output = brain.generateMotorOutput(decision_input);
    
    // Validate output
    if (!validateOutput(motor_output, 32, -5.0f, 5.0f)) { // Expected action size
        return false;
    }
    
    std::cout << "Motor output: " << decision_input.size() 
              << " decisions -> " << motor_output.size() << " actions" << std::endl;
    
    return true;
}

bool testAttentionSystem() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Generate test context
    auto context = generateRandomInput(2056); // Context size
    
    // Update attention
    brain.updateAttention(context);
    
    // Get attention weights
    auto attention_weights = brain.getAttentionWeights();
    
    // Validate attention weights
    if (attention_weights.empty()) {
        std::cout << "No attention weights returned" << std::endl;
        return false;
    }
    
    float total_weight = 0.0f;
    for (const auto& pair : attention_weights) {
        if (pair.second < 0.0f || pair.second > 2.0f) {
            std::cout << "Invalid attention weight for " << pair.first 
                      << ": " << pair.second << std::endl;
            return false;
        }
        total_weight += pair.second;
    }
    
    std::cout << "Attention system: " << attention_weights.size() 
              << " modules, total weight: " << total_weight << std::endl;
    
    return true;
}

bool testLearningAndAdaptation() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Apply learning signals
    float reward = 1.0f;
    float prediction_error = 0.5f;
    
    brain.applyLearning(reward, prediction_error);
    
    // Get learning statistics
    auto learning_stats = brain.getLearningStats();
    
    if (learning_stats.empty()) {
        std::cout << "No learning statistics returned" << std::endl;
        return false;
    }
    
    std::cout << "Learning stats for " << learning_stats.size() << " modules:" << std::endl;
    for (const auto& module_stats : learning_stats) {
        std::cout << "  " << module_stats.first << ": " 
                  << module_stats.second.size() << " metrics" << std::endl;
    }
    
    return true;
}

bool testPerformanceMetrics() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Update brain for several steps
    for (int i = 0; i < 10; ++i) {
        brain.update(0.1f);
    }
    
    // Get performance metrics
    auto metrics = brain.getPerformanceMetrics();
    
    if (metrics.empty()) {
        std::cout << "No performance metrics returned" << std::endl;
        return false;
    }
    
    // Validate key metrics
    std::vector<std::string> expected_metrics = {
        "total_activity", "update_count", "global_reward", 
        "num_modules", "num_connections"
    };
    
    for (const auto& metric : expected_metrics) {
        if (metrics.find(metric) == metrics.end()) {
            std::cout << "Missing expected metric: " << metric << std::endl;
            return false;
        }
    }
    
    std::cout << "Performance metrics:" << std::endl;
    for (const auto& pair : metrics) {
        std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    }
    
    return true;
}

bool testStabilityAndRobustness() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Test with various input sizes and values
    std::vector<std::pair<int, int>> test_resolutions = {
        {800, 600}, {1280, 720}, {1920, 1080}, {2560, 1440}
    };
    
    for (const auto& res : test_resolutions) {
        BrainModuleArchitecture test_brain;
        if (!test_brain.initialize(res.first, res.second)) {
            std::cout << "Failed to initialize with resolution " 
                      << res.first << "x" << res.second << std::endl;
            return false;
        }
        
        // Test processing with this resolution
        auto visual_input = generateVisualInput(res.first, res.second);
        auto features = test_brain.processVisualInput(visual_input);
        
        if (features.empty()) {
            std::cout << "Empty output for resolution " 
                      << res.first << "x" << res.second << std::endl;
            return false;
        }
    }
    
    // Test stability check
    if (!brain.isStable()) {
        std::cout << "Brain architecture reported as unstable" << std::endl;
        return false;
    }
    
    std::cout << "Stability test passed for " << test_resolutions.size() 
              << " resolutions" << std::endl;
    
    return true;
}

// ============================================================================
// AUTONOMOUS LEARNING AGENT TESTS
// ============================================================================

bool testAutonomousAgentInitialization() {
    NetworkConfig config;
    config.input_size = 64;
    config.hidden_size = 256;
    config.output_size = 32;
    config.finalizeConfig();
    
    AutonomousLearningAgent agent(config);
    
    if (!agent.initialize()) {
        std::cout << "Failed to initialize autonomous agent" << std::endl;
        return false;
    }
    
    std::cout << "Autonomous agent initialized successfully" << std::endl;
    return true;
}

bool testAutonomousLearningLoop() {
    NetworkConfig config;
    config.input_size = 64;
    config.hidden_size = 256;
    config.output_size = 32;
    config.finalizeConfig();
    
    AutonomousLearningAgent agent(config);
    if (!agent.initialize()) return false;
    
    // Start learning
    agent.startAutonomousLearning();
    
    // Run learning steps
    float total_progress = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float progress = agent.autonomousLearningStep(0.1f);
        total_progress += progress;
        
        if (progress < 0.0f || progress > 1.0f) {
            std::cout << "Invalid progress value: " << progress << std::endl;
            return false;
        }
    }
    
    // Stop learning
    agent.stopAutonomousLearning();
    
    std::cout << "Learning loop completed, average progress: " 
              << (total_progress / 10.0f) << std::endl;
    
    return true;
}

bool testAgentStatusReporting() {
    NetworkConfig config;
    config.input_size = 64;
    config.hidden_size = 256;
    config.output_size = 32;
    config.finalizeConfig();
    
    AutonomousLearningAgent agent(config);
    if (!agent.initialize()) return false;
    
    // Get status report
    auto status = agent.getStatusReport();
    if (status.empty()) {
        std::cout << "Empty status report" << std::endl;
        return false;
    }
    
    // Get learning progress
    float progress = agent.getLearningProgress();
    if (progress < 0.0f || progress > 1.0f) {
        std::cout << "Invalid learning progress: " << progress << std::endl;
        return false;
    }
    
    // Get attention weights
    auto attention = agent.getAttentionWeights();
    if (attention.empty()) {
        std::cout << "No attention weights returned" << std::endl;
        return false;
    }
    
    std::cout << "Status report length: " << status.length() << " chars" << std::endl;
    std::cout << "Learning progress: " << progress << std::endl;
    std::cout << "Attention modules: " << attention.size() << std::endl;
    
    return true;
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

bool testFullPipelineIntegration() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Simulate complete processing pipeline
    auto visual_input = generateVisualInput(1920, 1080);
    auto text_input = generateRandomInput(256);
    auto goals = generateRandomInput(64);
    
    // Step 1: Visual processing
    auto visual_features = brain.processVisualInput(visual_input);
    if (visual_features.empty()) return false;
    
    // Step 2: Text comprehension
    auto comprehension_output = brain.processTextInput(text_input);
    if (comprehension_output.empty()) return false;
    
    // Step 3: Decision making
    auto decisions = brain.executeDecisionMaking(visual_features, goals);
    if (decisions.empty()) return false;
    
    // Step 4: Motor output
    auto actions = brain.generateMotorOutput(decisions);
    if (actions.empty()) return false;
    
    // Step 5: Learning
    brain.applyLearning(1.0f, 0.5f);
    
    // Step 6: Update
    brain.update(0.1f);
    
    std::cout << "Full pipeline: visual(" << visual_input.size() 
              << ") -> features(" << visual_features.size()
              << ") -> decisions(" << decisions.size()
              << ") -> actions(" << actions.size() << ")" << std::endl;
    
    return true;
}

bool testMemoryOperations() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    // Store experiences
    for (int i = 0; i < 5; ++i) {
        auto experience = generateRandomInput(512);
        brain.storeExperience(experience, "test_context_" + std::to_string(i));
    }
    
    // Retrieve experiences
    auto query = generateRandomInput(512);
    auto retrieved = brain.retrieveExperiences(query, 3);
    
    if (retrieved.empty()) {
        std::cout << "No experiences retrieved" << std::endl;
        return false;
    }
    
    std::cout << "Memory operations: stored 5, retrieved " 
              << retrieved.size() << " experiences" << std::endl;
    
    return true;
}

// ============================================================================
// PERFORMANCE BENCHMARKS
// ============================================================================

bool benchmarkProcessingSpeed() {
    BrainModuleArchitecture brain;
    if (!brain.initialize(1920, 1080)) return false;
    
    const int num_iterations = 100;
    auto visual_input = generateVisualInput(1920, 1080);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto features = brain.processVisualInput(visual_input);
        brain.update(0.01f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    double avg_time = duration / num_iterations;
    double fps = 1000.0 / avg_time;
    
    std::cout << "Processing speed: " << std::fixed << std::setprecision(2)
              << avg_time << "ms per frame (" << fps << " FPS)" << std::endl;
    
    // Performance threshold: should process at least 10 FPS
    return fps >= 10.0;
}

bool benchmarkMemoryUsage() {
    // This is a simplified memory usage test
    // In a real implementation, you would use system-specific memory monitoring
    
    std::vector<std::unique_ptr<BrainModuleArchitecture>> brains;
    
    // Create multiple brain instances
    for (int i = 0; i < 5; ++i) {
        auto brain = std::make_unique<BrainModuleArchitecture>();
        if (!brain->initialize(1920, 1080)) {
            std::cout << "Failed to create brain instance " << i << std::endl;
            return false;
        }
        brains.push_back(std::move(brain));
    }
    
    std::cout << "Memory usage test: created " << brains.size() 
              << " brain instances successfully" << std::endl;
    
    return true;
}

// ============================================================================
// MAIN TEST EXECUTION
// ============================================================================

int main() {
    std::cout << "=== BRAIN MODULE ARCHITECTURE TEST SUITE ===" << std::endl;
    std::cout << "Testing comprehensive modular agent functionality" << std::endl;
    
    TestFramework framework;
    
    // Core Architecture Tests
    framework.runTest("Brain Architecture Initialization", testBrainArchitectureInitialization);
    framework.runTest("Visual Processing Pipeline", testVisualProcessingPipeline);
    framework.runTest("Decision Making Pipeline", testDecisionMakingPipeline);
    framework.runTest("Motor Output Generation", testMotorOutputGeneration);
    framework.runTest("Attention System", testAttentionSystem);
    framework.runTest("Learning and Adaptation", testLearningAndAdaptation);
    framework.runTest("Performance Metrics", testPerformanceMetrics);
    framework.runTest("Stability and Robustness", testStabilityAndRobustness);
    
    // Autonomous Agent Tests
    framework.runTest("Autonomous Agent Initialization", testAutonomousAgentInitialization);
    framework.runTest("Autonomous Learning Loop", testAutonomousLearningLoop);
    framework.runTest("Agent Status Reporting", testAgentStatusReporting);
    
    // Integration Tests
    framework.runTest("Full Pipeline Integration", testFullPipelineIntegration);
    framework.runTest("Memory Operations", testMemoryOperations);
    
    // Performance Benchmarks
    framework.runTest("Processing Speed Benchmark", benchmarkProcessingSpeed);
    framework.runTest("Memory Usage Benchmark", benchmarkMemoryUsage);
    
    // Print comprehensive summary
    framework.printSummary();
    
    return 0;
}
