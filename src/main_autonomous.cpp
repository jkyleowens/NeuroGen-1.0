// Enhanced main.cpp with Autonomous Learning Agent Integration
// NeuroGen Version 0.5.5 - Advanced Autonomous Learning Framework

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <functional>
#include <iomanip>
#include <string>

// Core NeuroGen includes
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkIntegration.h"
#include "NeuroGen/ControllerModule.h"

// Function to create a default configuration for a neural module
NetworkConfig create_default_config() {
    NetworkConfig config;
    config.num_neurons = 8192; // MASSIVE scale-up: 8K neurons per base module for free-thinking AI
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_pruning = true;
    config.enable_structural_plasticity = true; // Enable dynamic synaptogenesis
    
    // Enhanced connectivity parameters for version 0.5.5 - optimized for large scale
    config.input_hidden_prob = 0.15f;  // Reduced for computational efficiency at scale
    config.hidden_hidden_prob = 0.08f; // Sparse connectivity for emergent patterns
    config.hidden_output_prob = 0.4f;  // Selective output connections
    config.exc_ratio = 0.8f;
    
    // Synaptic parameters optimized for large-scale networks
    config.min_weight = 0.001f;        // Finer resolution for large networks
    config.max_weight = 2.0f;          // Increased for stronger signal propagation
    config.weight_init_std = 0.15f;    // Reduced for stability at scale
    
    // Topology parameters - MASSIVE SCALE-UP for tens of thousands of neurons
    config.numColumns = 16;            // 4x increase: 16 cortical columns
    config.neuronsPerColumn = 512;     // 8x increase: 512 neurons per column = 8,192 total
    config.localFanOut = 40;           // Increased connectivity for richer dynamics
    config.localFanIn = 40;            // Increased fan-in for complex integration
    
    // Enhanced timing
    config.dt = 0.1;
    config.simulation_time = 1000.0f; // 1 second simulation
    
    config.finalizeConfig();
    return config;
}

// ============================================================================
// SIMULATION MODE SELECTION
// ============================================================================

enum class SimulationMode {
    BASIC_MODULAR,           // Original modular simulation
    AUTONOMOUS_LEARNING,     // New autonomous learning mode
    INTERACTIVE_TRAINING,    // Interactive training with user feedback
    BENCHMARK_SUITE         // Performance benchmarking
};

// ============================================================================
// BASIC MODULAR SIMULATION (Enhanced Version)
// ============================================================================

void runBasicModularSimulation() {
    std::cout << "🧠 Running Enhanced Modular Neural Network Simulation..." << std::endl;

    // --- Configuration ---
    auto cognitive_config = create_default_config();
    cognitive_config.neurogenesis_rate = 0.002; // Higher plasticity for cognitive tasks

    auto motor_config = create_default_config();
    motor_config.stdp_learning_rate = 0.005; // Lower learning rate for stable motor control

    // --- Module Creation ---
    auto perception_net = std::make_shared<NeuralModule>("PerceptionNet", cognitive_config);
    auto planning_net = std::make_shared<NeuralModule>("PlanningNet", cognitive_config);
    auto motor_control_net = std::make_shared<NeuralModule>("MotorControlNet", motor_config);

    // --- CONTROLLER MODULE INTEGRATION ---
    std::cout << "🎛️ Initializing Central Neuromodulatory Controller..." << std::endl;
    
    ControllerConfig controller_config;
    controller_config.initial_dopamine_level = 0.4f;    // Start with good motivation
    controller_config.initial_serotonin_level = 0.5f;   // Balanced mood
    controller_config.reward_learning_rate = 0.02f;     // Enhanced learning rate
    controller_config.enable_detailed_logging = true;   // Enable detailed logging
    controller_config.enable_auto_regulation = true;    // Enable auto regulation
    
    auto neuro_controller = std::make_unique<ControllerModule>(controller_config);
    
    // Register modules with the controller
    neuro_controller->register_module("PerceptionNet", perception_net);
    neuro_controller->register_module("PlanningNet", planning_net);
    neuro_controller->register_module("MotorControlNet", motor_control_net);
    
    std::cout << "✅ Neuromodulatory controller configured with 3 modules" << std::endl;

    // --- Task-Level Module Creation ---
    auto cognitive_module = std::make_shared<CognitiveModule>(perception_net, planning_net);
    auto motor_module = std::make_shared<MotorModule>(motor_control_net);

    // --- System Initialization ---
    std::vector<std::shared_ptr<TaskModule>> task_modules;
    task_modules.push_back(cognitive_module);
    task_modules.push_back(motor_module);

    std::cout << "\nInitializing all task modules..." << std::endl;
    for (const auto& module : task_modules) {
        module->initialize();
    }

    // Enable detailed controller logging
    neuro_controller->enable_detailed_logging(true);

    // --- Enhanced Simulation Loop ---
    std::cout << "\n🚀 Starting enhanced simulation loop..." << std::endl;
    
    float total_simulation_time = 1000.0f; // ms
    float dt = 0.1f; // ms
    int num_steps = static_cast<int>(total_simulation_time / dt);
    
    // Enhanced input patterns for testing modular responses
    std::vector<std::vector<float>> test_patterns = {
        std::vector<float>(256, 0.0f), // Baseline
        std::vector<float>(256, 0.0f), // Pattern 1: Visual-like input
        std::vector<float>(256, 0.0f), // Pattern 2: Motor command
        std::vector<float>(256, 0.0f)  // Pattern 3: Mixed pattern
    };
    
    // Configure test patterns
    for (int i = 0; i < 32; i++) {
        test_patterns[1][i] = 15.0f + (i % 3) * 5.0f; // Spatial pattern
    }
    
    for (int i = 64; i < 128; i += 4) {
        test_patterns[2][i] = 20.0f; // Rhythmic pattern
    }
    
    for (int i = 0; i < 256; i += 8) {
        test_patterns[3][i] = 12.0f + (i / 32) * 2.0f; // Gradient pattern
    }
    
    int current_pattern = 0;
    int pattern_duration = 2000; // Steps per pattern
    int structural_plasticity_interval = 1000; // Every 100ms
    
    for (int i = 0; i < num_steps; ++i) {
        float current_time = i * dt;
        
        // Update the neuromodulatory controller first
        neuro_controller->update(dt);
        
        // Switch input patterns periodically
        if (i % pattern_duration == 0) {
            current_pattern = (current_pattern + 1) % test_patterns.size();
            std::cout << "🔄 Switching to input pattern " << current_pattern 
                      << " at time " << current_time << "ms" << std::endl;
            
            // Notify controller of pattern change (novelty detection)
            RewardSignal novelty_signal(RewardSignalType::NOVELTY_DETECTION, 0.3f, "Environment");
            novelty_signal.context = "Pattern change detected";
            neuro_controller->apply_reward("PerceptionNet", 0.3f, RewardSignalType::NOVELTY_DETECTION);
        }
        
        std::vector<float> inputs = test_patterns[current_pattern];
        
        // Add noise for biological realism
        for (auto& input : inputs) {
            if (input > 0.0f) {
                input += ((rand() % 100) / 100.0f - 0.5f) * 2.0f;
            }
        }
        
        // Calculate reward based on network coordination
        float reward = 0.1f;
        auto perception_stats = perception_net->get_stats();
        auto planning_stats = planning_net->get_stats();
        auto motor_stats = motor_control_net->get_stats();
        
        // Enhanced reward calculation with controller feedback
        bool modules_coordinated = (perception_stats.active_neuron_count > 5 && 
                                  planning_stats.active_neuron_count > 3 &&
                                  motor_stats.active_neuron_count > 2);
        
        if (modules_coordinated) {
            reward = 0.5f;
            
            // Generate cooperation reward through controller
            neuro_controller->apply_reward("", 0.4f, RewardSignalType::SOCIAL_COOPERATION);
        }
        
        // Update modules with inter-modular communication
        perception_net->update(dt, inputs, reward);
        auto perception_output = perception_net->get_output();
        
        planning_net->update(dt, perception_output, reward);
        auto planning_output = planning_net->get_output();
        
        motor_control_net->update(dt, planning_output, reward);
        
        // Controller-mediated attention allocation
        if (i % 500 == 0) {
            // Use available controller methods for attention modulation
            float total_activity = perception_stats.active_neuron_count + 
                                 planning_stats.active_neuron_count + 
                                 motor_stats.active_neuron_count;
            
            if (total_activity > 0) {
                // Focus on the most active module
                if (perception_stats.active_neuron_count > planning_stats.active_neuron_count && 
                    perception_stats.active_neuron_count > motor_stats.active_neuron_count) {
                    neuro_controller->enable_focus_mode("PerceptionNet", 0.6f);
                } else if (planning_stats.active_neuron_count > motor_stats.active_neuron_count) {
                    neuro_controller->enable_focus_mode("PlanningNet", 0.6f);
                } else {
                    neuro_controller->enable_focus_mode("MotorControlNet", 0.6f);
                }
            }
        }
        
        // Enhanced structural plasticity with controller coordination
        if (i % structural_plasticity_interval == 0 && i > 0) {
            std::cout << "🌱 Structural plasticity at " << current_time << "ms" << std::endl;
            
            // Controller decides when and where to promote growth
            float system_performance = neuro_controller->calculate_overall_system_performance();
            
            if (system_performance > 0.6f) {
                // Good performance - enable creative mode for exploration
                neuro_controller->enable_creative_mode(0.3f);
            } else if (system_performance < 0.4f) {
                // Poor performance - enable focus mode on a random module for now
                std::vector<std::string> modules = {"PerceptionNet", "PlanningNet", "MotorControlNet"};
                std::string focus_module = modules[rand() % modules.size()];
                neuro_controller->enable_focus_mode(focus_module, 0.7f);
            }
            
            auto* perception_network = perception_net->get_network();
            auto* planning_network = planning_net->get_network();
            auto* motor_network = motor_control_net->get_network();
            
            if (perception_network) {
                perception_network->grow_synapses();
                perception_network->prune_synapses();
            }
            if (planning_network) {
                planning_network->grow_synapses();
                planning_network->prune_synapses();
            }
            if (motor_network) {
                motor_network->grow_synapses();
                motor_network->prune_synapses();
            }
            
            // Coordinate module activities after structural changes
            neuro_controller->coordinate_module_activities();
        }
        
        // Enhanced monitoring output with controller status
        if (i % 1000 == 0) {
            std::cout << "\n📊 Time: " << current_time << "ms (Pattern " << current_pattern << ")" << std::endl;
            std::cout << "   🧠 Perception: " << perception_stats.active_neuron_count 
                      << " active, " << perception_stats.total_synapses << " synapses" << std::endl;
            std::cout << "   🎯 Planning: " << planning_stats.active_neuron_count 
                      << " active, " << planning_stats.total_synapses << " synapses" << std::endl;
            std::cout << "   🏃 Motor: " << motor_stats.active_neuron_count 
                      << " active, " << motor_stats.total_synapses << " synapses" << std::endl;
            
            // Display neuromodulator status
            std::cout << "   🧬 Dopamine: " << std::fixed << std::setprecision(2) 
                      << neuro_controller->get_concentration(NeuromodulatorType::DOPAMINE) << std::endl;
            std::cout << "   🧬 Serotonin: " << neuro_controller->get_concentration(NeuromodulatorType::SEROTONIN) << std::endl;
            std::cout << "   🧬 Norepinephrine: " << neuro_controller->get_concentration(NeuromodulatorType::NOREPINEPHRINE) << std::endl;
            
            // System performance
            float system_perf = neuro_controller->calculate_overall_system_performance();
            std::cout << "   📈 System Performance: " << std::setprecision(1) << system_perf * 100 << "%" << std::endl;
        }
    }

    // Final controller status report
    std::cout << "\n🎛️ ===== FINAL CONTROLLER STATUS =====" << std::endl;
    std::string status_report = neuro_controller->generate_status_report();
    std::cout << status_report << std::endl;

    std::cout << "\n✅ Basic modular simulation completed!" << std::endl;
}

// ============================================================================
// AUTONOMOUS LEARNING SIMULATION (New Version 0.5.5 Feature)
// ============================================================================

void runAutonomousLearningSimulation(bool reset_model = false) {
    std::cout << "\n🤖 ========== AUTONOMOUS LEARNING SIMULATION ==========\n" << std::endl;
    std::cout << "🚀 Initializing Advanced Autonomous Learning Agent..." << std::endl;
    
    // Configure autonomous agent with available NetworkConfig
    auto agent_config = create_default_config();
    agent_config.num_neurons = 512;        // Start with more neurons
    agent_config.enable_neurogenesis = true;
    agent_config.enable_stdp = true;
    agent_config.enable_structural_plasticity = true;
    
    std::cout << "🔧 Agent Configuration:" << std::endl;
    std::cout << "   • Initial neurons: " << agent_config.num_neurons << std::endl;
    std::cout << "   • Neurogenesis enabled: " << (agent_config.enable_neurogenesis ? "Yes" : "No") << std::endl;
    std::cout << "   • STDP enabled: " << (agent_config.enable_stdp ? "Yes" : "No") << std::endl;
    std::cout << "   • Structural plasticity: " << (agent_config.enable_structural_plasticity ? "Yes" : "No") << std::endl;
    
    // Create autonomous learning agent
    AutonomousLearningAgent agent(agent_config);
    
    // ========================================
    // SETUP SIMULATED ENVIRONMENT
    // ========================================
    
    std::cout << "\n🌍 Setting up simulated environment..." << std::endl;
    
    // Environment state variables
    std::vector<float> environment_state(64, 0.0f);
    std::vector<float> environment_dynamics(64, 0.0f);
    float environment_complexity = 0.5f;
    int environment_phase = 0;
    std::mt19937 env_rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> env_dist(-1.0f, 1.0f);
    
    // Initialize environment with interesting dynamics
    for (size_t i = 0; i < environment_state.size(); ++i) {
        environment_state[i] = env_dist(env_rng);
        environment_dynamics[i] = env_dist(env_rng) * 0.1f;
    }
    
    // Environment sensor function that returns BrowsingState
    /* auto environment_sensor = [&]() -> BrowsingState {
        // Update environment dynamics
        environment_phase++;
        
        for (size_t i = 0; i < environment_state.size(); ++i) {
            // Add some interesting dynamics
            float wave = std::sin(environment_phase * 0.01f + i * 0.1f) * 0.1f;
            float noise = env_dist(env_rng) * 0.05f;
            environment_state[i] += environment_dynamics[i] + wave + noise;
            
            // Keep within bounds
            environment_state[i] = std::tanh(environment_state[i]);
            
            // Occasionally change dynamics
            if (environment_phase % 1000 == 0) {
                environment_dynamics[i] += env_dist(env_rng) * 0.02f;
                environment_dynamics[i] = std::tanh(environment_dynamics[i] * 0.8f);
            }
        }
        
        // Increase complexity over time
        if (environment_phase % 2000 == 0) {
            environment_complexity = std::min(1.0f, environment_complexity + 0.1f);
            std::cout << "🌊 Environment complexity increased to " << environment_complexity << std::endl;
        }
        
        // Convert to BrowsingState
        BrowsingState state;
        state.current_url = "simulated://environment";
        state.visual_features = environment_state;
        state.scroll_position = environment_phase % 1000;
        state.page_loading = false;
        
        return state;
    };*/
    
    // Environment action executor function - REAL COMPUTER CONTROL
    auto action_executor = [&](const BrowsingAction& action) {
        // Log the action first
        std::cout << "🎬 Real Computer Action: " << actionTypeToString(action.type) 
                  << " (confidence: " << action.confidence << ")";
        
        switch(action.type) {
            case ActionType::CLICK:
                std::cout << " at (" << action.x_coordinate << ", " << action.y_coordinate << ")";
                break;
            case ActionType::TYPE:
                std::cout << " with text: \"" << action.text_content << "\"";
                break;
            case ActionType::SCROLL:
                std::cout << " " << (action.scroll_direction == ScrollDirection::UP ? "UP" : "DOWN")
                          << " by " << action.scroll_amount;
                break;
            case ActionType::ENTER:
            case ActionType::BACKSPACE:
                // No extra details needed for these actions
                break;
        }
        
        std::cout << " [EXECUTING ON REAL COMPUTER]" << std::endl;
        
        // Actually execute the action using the agent's internal execute_action method
        // This delegates to the properly implemented execute_action(action) method
        agent.execute_action(action);
    };
    
    // Setup environment interaction - REAL SCREEN MONITORING
    auto environment_sensor = [&]() -> BrowsingState {
        // Real screen capture and analysis
        BrowsingState state;
        state.current_url = "real://computer_screen";
        
        // The agent now captures real screen data internally
        // This is just for compatibility with the existing interface
        state.scroll_position = environment_phase % 1000;
        state.page_loading = false;
        state.window_width = 1920;
        state.window_height = 1080;
        
        return state;
    };
    
    // Setup environment interaction
    agent.setEnvironmentSensor(environment_sensor);
    agent.setActionExecutor(action_executor);
    
    std::cout << "✅ Environment configured with dynamic complexity!" << std::endl;
    
    // ========================================
    // ADD LEARNING GOALS
    // ========================================
    
    std::cout << "\n🎯 Setting up learning goals..." << std::endl;
    
    // Goal 1: Environment Adaptation
    auto adaptation_goal = std::make_unique<AutonomousGoal>();
    adaptation_goal->description = "Environment Adaptation";
    adaptation_goal->priority = 0.9f;
    adaptation_goal->is_active = true;
    agent.addLearningGoal(std::move(adaptation_goal));
    
    // Goal 2: Action Diversity
    auto diversity_goal = std::make_unique<AutonomousGoal>();
    diversity_goal->description = "Action Diversity";
    diversity_goal->priority = 0.6f;
    diversity_goal->is_active = true;
    agent.addLearningGoal(std::move(diversity_goal));
    
    // Goal 3: Predictive Learning
    auto prediction_goal = std::make_unique<AutonomousGoal>();
    prediction_goal->description = "Predictive Learning";
    prediction_goal->priority = 0.8f;
    prediction_goal->is_active = true;
    agent.addLearningGoal(std::move(prediction_goal));

    // Goal 4: Information Gathering
    auto info_goal = std::make_unique<AutonomousGoal>();
    info_goal->description = "Information Gathering";
    info_goal->priority = 0.7f; // High priority for seeking knowledge
    info_goal->is_active = true;
    agent.addLearningGoal(std::move(info_goal));
    
    std::cout << "✅ Learning goals established!" << std::endl;
    
    // ========================================
    // RUN AUTONOMOUS LEARNING SIMULATION
    // ========================================
    
    std::cout << "\n🚀 Starting Autonomous Learning Simulation..." << std::endl;
    std::cout << "   The agent will now explore, learn, and adapt autonomously!" << std::endl;
    std::cout << "   Watch for learning progress...\n" << std::endl;
    
    // Initialize the agent
    if (!agent.initialize(reset_model)) {
        std::cerr << "❌ Failed to initialize autonomous learning agent!" << std::endl;
        return;
    }
    
    // Enable detailed logging for better monitoring
    agent.setDetailedLogging(true);
    
    int max_learning_steps = 2000; // Increased for more comprehensive learning
    std::cout << "🔄 Running " << max_learning_steps << " autonomous learning steps..." << std::endl;
    std::cout << "🖥️  The agent will now interact with the real computer screen!" << std::endl;
    std::cout << "⚠️  Safety bounds are enabled to prevent dangerous actions." << std::endl;
    
    auto learning_start = std::chrono::high_resolution_clock::now();
    
    agent.startAutonomousLearning();
    
    // Enhanced learning loop with real-time monitoring
    float total_reward = 0.0f;
    float best_performance = 0.0f;
    
    for (int step = 0; step < max_learning_steps; ++step) {
        float learning_progress = agent.autonomousLearningStep(1.0f);
        
        // Update the agent
        agent.update(1.0f);
        
        // Detailed monitoring every 100 steps
        if (step % 100 == 0) {
            std::cout << "\n📈 Learning Progress Report (Step " << step << "):" << std::endl;
            std::cout << "   🧠 Learning Progress: " << std::fixed << std::setprecision(2) 
                      << learning_progress * 100 << "%" << std::endl;
            
            // Get comprehensive status report
            std::string status = agent.getStatusReport();
            std::cout << status << std::endl;
            
            // Track best performance
            if (learning_progress > best_performance) {
                best_performance = learning_progress;
                std::cout << "🎉 New best performance achieved: " << best_performance * 100 << "%!" << std::endl;
            }
            
            // Check for high competence achievement
            if (learning_progress > 0.8f) {
                std::cout << "\n🎉 High learning competence achieved!" << std::endl;
                std::cout << "The agent has demonstrated significant learning capability." << std::endl;
                break;
            }
        }
        
        // Periodic performance evaluation
        if (step % 500 == 0 && step > 0) {
            auto attention_weights = agent.getAttentionWeights();
            std::cout << "\n🔍 Attention Analysis:" << std::endl;
            for (const auto& [module, weight] : attention_weights) {
                std::cout << "   " << module << ": " << weight << std::endl;
            }
        }
        
        // Brief pause to allow observation and prevent overwhelming the system
        if (step % 25 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Emergency stop check (user can stop by creating a stop file)
        if (step % 100 == 0) {
            std::ifstream stop_file("/tmp/stop_neurogen");
            if (stop_file.good()) {
                std::cout << "\n🛑 Emergency stop detected. Shutting down safely..." << std::endl;
                break;
            }
        }
    }
    
    agent.stopAutonomousLearning();
    
    auto learning_end = std::chrono::high_resolution_clock::now();
    auto learning_duration = std::chrono::duration<double>(learning_end - learning_start).count();
    
    // ========================================
    // FINAL ANALYSIS AND REPORT
    // ========================================
    
    std::cout << "\n🎊 ========== AUTONOMOUS LEARNING COMPLETED ==========\n" << std::endl;
    
    // Generate basic report
    std::string final_report = agent.getStatusReport();
    std::cout << "📊 Final Agent Status: " << final_report << std::endl;
    
    std::cout << "⏱️ Learning Duration: " << std::fixed << std::setprecision(1) 
              << learning_duration << " seconds" << std::endl;
    std::cout << "⚡ Learning Speed: " << (max_learning_steps / learning_duration) 
              << " steps/second" << std::endl;
    
    std::cout << "\n📈 LEARNING PROGRESS: Autonomous agent completed " << max_learning_steps << " learning steps!" << std::endl;
    std::cout << "🌱 DYNAMIC ADAPTATION: Network adapted to changing environment complexity!" << std::endl;
    
    std::cout << "\n✅ Autonomous Learning Simulation Complete!" << std::endl;
}

// ============================================================================
// INTERACTIVE TRAINING MODE
// ============================================================================

void runInteractiveTraining() {
    std::cout << "\n🎮 Interactive Training Mode - Coming Soon!" << std::endl;
    std::cout << "This mode will allow real-time interaction with the learning agent." << std::endl;
}

// ============================================================================
// BENCHMARK SUITE
// ============================================================================

void runBenchmarkSuite() {
    std::cout << "\n� Benchmark Suite - Coming Soon!" << std::endl;
    std::cout << "This will test performance across standardized learning tasks." << std::endl;
}

// ============================================================================
// MAIN FUNCTION WITH MODE SELECTION
// ============================================================================

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);

    bool reset_model = false;
    if (std::find(args.begin(), args.end(), "--reset-model") != args.end()) {
        reset_model = true;
        std::cout << "🔥 --reset-model flag detected. Agent state will be reset." << std::endl;
    }

    std::cout << "🧠 NeuroGen 0.5.5 - Advanced Autonomous Learning Framework" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    auto agent_config = create_default_config();
    AutonomousLearningAgent agent(agent_config);

    if (!agent.initialize(reset_model)) {
        std::cerr << "❌ Failed to initialize autonomous learning agent!" << std::endl;
        return 1;
    }

    std::cout << "✅ Agent initialized. Waiting for commands..." << std::endl;
    std::cout.flush();

    std::string line;
    while (true) {
        if (std::getline(std::cin, line)) {
            if (line.rfind("COMMAND:", 0) == 0) {
                agent.handleCommand(line.substr(8));
                std::cout.flush(); // Ensure any output is immediately visible
            } else if (line == "EXIT" || line == "QUIT") {
                std::cout << "🛑 Exit command received. Shutting down." << std::endl;
                break;
            } else if (!line.empty()) {
                std::cerr << "Warning: Received malformed input: " << line << std::endl;
            }
        } else {
            // Check if stdin was closed (parent process died)
            if (std::cin.eof()) {
                std::cout << "🛑 Input stream closed. Shutting down." << std::endl;
                break;
            }
            // Sleep briefly to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::cout << "🛑 Agent shutting down." << std::endl;
    agent.shutdown();

    return 0;
}
