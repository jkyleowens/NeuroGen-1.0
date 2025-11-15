// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION
// File: src/AutonomousLearningAgent.cpp
// ============================================================================

#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/SafetyManager.h"
#include "NeuroGen/AttentionController.h"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION
// ============================================================================
AutonomousLearningAgent::AutonomousLearningAgent(const NetworkConfig& config)
    : config_(config), gen(std::random_device{}()), save_path_("neural_agent_saves") {
    controller_module_ = std::make_unique<ControllerModule>();
    memory_system_ = std::make_unique<MemorySystem>(10000, 512);
    attention_controller_ = std::make_unique<AttentionController>();
    input_controller_ = std::make_unique<InputController>();
    brain_architecture_ = std::make_unique<BrainModuleArchitecture>();

    // Initialize environmental context and global state
    environmental_context_.resize(1024, 0.0f);
    global_state_.resize(2048, 0.0f);
    global_reward_signal_ = 0.0f;
    exploration_rate_ = 0.9f; // Start with high exploration
    is_learning_active_ = true;
    detailed_logging_ = false;
    is_passive_mode_ = false;
    simulation_time_ = 0.0f;
    last_action_time_ = std::chrono::steady_clock::now();
}

AutonomousLearningAgent::~AutonomousLearningAgent() {
    // Destructor logic
}

void AutonomousLearningAgent::initializeSpecializedModules() {
    // Create specialized neural modules for different cognitive functions
    // LAPTOP-FRIENDLY CONFIGURATION: Reduced scale for gaming laptop (ASUS TUF)
    // Total neurons: ~5K (down from ~35K) for better performance on consumer hardware

    // Prefrontal Cortex - Executive function and reasoning (2,048 neurons)
    auto prefrontal_cortex_config = config_;
    prefrontal_cortex_config.num_neurons = 2048;   // Reduced to 2K neurons
    prefrontal_cortex_config.numColumns = 8;       // 8 executive columns
    prefrontal_cortex_config.neuronsPerColumn = 256;
    prefrontal_cortex_config.localFanOut = 40;     // Reduced connectivity for efficiency
    modules_["prefrontal_cortex"] = std::make_unique<SpecializedModule>("prefrontal_cortex", prefrontal_cortex_config);

    // Motor Cortex - Precise motor control (1,024 neurons)
    auto motor_cortex_config = config_;
    motor_cortex_config.num_neurons = 1024;        // Reduced to 1K neurons
    motor_cortex_config.numColumns = 4;            // 4 motor columns
    motor_cortex_config.neuronsPerColumn = 256;
    motor_cortex_config.localFanOut = 25;          // Reduced connectivity
    modules_["motor_cortex"] = std::make_unique<SpecializedModule>("motor_cortex", motor_cortex_config);

    // Working Memory - Short-term memory and manipulation (1,024 neurons)
    auto working_memory_config = config_;
    working_memory_config.num_neurons = 1024;      // Reduced to 1K neurons
    working_memory_config.numColumns = 4;          // 4 memory columns
    working_memory_config.neuronsPerColumn = 256;
    working_memory_config.localFanOut = 20;
    modules_["working_memory"] = std::make_unique<SpecializedModule>("working_memory", working_memory_config);

    // Reward System - Value estimation and reinforcement (512 neurons)
    auto reward_system_config = config_;
    reward_system_config.num_neurons = 512;        // Reduced to 512 neurons
    reward_system_config.numColumns = 2;           // 2 reward columns
    reward_system_config.neuronsPerColumn = 256;
    reward_system_config.localFanOut = 15;
    modules_["reward_system"] = std::make_unique<SpecializedModule>("reward_system", reward_system_config);

    // Attention System - Dynamic focus and resource allocation (512 neurons)
    auto attention_system_config = config_;
    attention_system_config.num_neurons = 512;     // Reduced to 512 neurons
    attention_system_config.numColumns = 2;        // 2 attention columns
    attention_system_config.neuronsPerColumn = 256;
    attention_system_config.localFanOut = 15;
    modules_["attention_system"] = std::make_unique<SpecializedModule>("attention_system", attention_system_config);
}

bool AutonomousLearningAgent::initialize(bool real_time_capture) {
    std::cout << "🤖 Initializing Autonomous Learning Agent..." << std::endl;

    // Initialize specialized modules
    initializeSpecializedModules();

    // Register neural modules with attention controller
    attention_controller_->register_module("motor_cortex");
    attention_controller_->register_module("prefrontal_cortex");
    attention_controller_->register_module("working_memory");
    attention_controller_->register_module("reward_system");
    attention_controller_->register_module("attention_system");

    // Initialize the safety manager with screen dimensions
    SafetyManager::getInstance().setScreenDimensions(1920, 1080);

    std::cout << "✅ Agent initialization complete." << std::endl;
    return true;
}

void AutonomousLearningAgent::shutdown() {
    std::cout << "🤖 Shutting down Autonomous Learning Agent..." << std::endl;
    // Shutdown logic here
    std::cout << "✅ Agent shutdown complete." << std::endl;
}

void AutonomousLearningAgent::start() {
    // Implementation for starting the agent
    current_mode_ = OperatingMode::AUTONOMOUS;
    std::cout << "Agent started." << std::endl;
}

void AutonomousLearningAgent::stop() {
    // Implementation for stopping the agent
    current_mode_ = OperatingMode::IDLE;
    std::cout << "Agent stopped." << std::endl;
}

void AutonomousLearningAgent::run() {
    // Main loop for the agent
    while (current_mode_ == OperatingMode::AUTONOMOUS) {
        autonomousLearningStep(0.1f); // Example time step
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void AutonomousLearningAgent::update(float dt) {
    simulation_time_ += dt;

    if (controller_module_) {
        controller_module_->update(dt);
    }

    if (is_learning_active_) {
        autonomousLearningStep(dt);
        update_learning_goals();
    }
}

void AutonomousLearningAgent::handleCommand(const std::string& command) {
    // Parse command type
    if (command.rfind("LANGUAGE_INPUT:", 0) == 0) {
        // Extract the text after "LANGUAGE_INPUT:"
        std::string input_text = command.substr(15);  // 15 = length of "LANGUAGE_INPUT:"
        
        // Process the language input
        if (processLanguageInput(input_text)) {
            // Generate response
            std::string response = generateLanguageResponse();
            
            // CRITICAL: Output the prediction to stdout
            std::cout << "NEXT_WORD_PREDICTION:" << response << std::endl;
            std::cout.flush();
        } else {
            std::cout << "NEXT_WORD_PREDICTION:" << std::endl;
            std::cout.flush();
        }
    }
    else if (command.rfind("SET_MODE:", 0) == 0) {
        std::string mode = command.substr(9);
        if (mode == "LANGUAGE_TRAINING") {
            std::cout << "✅ Language training mode activated" << std::endl;
            // Note: nlp_mode_active_ is already set to true in constructor
            // Just acknowledge the mode change
        }
    }
    else if (command.rfind("REWARD_SIGNAL:", 0) == 0) {
        try {
            float reward = std::stof(command.substr(14));
            global_reward_signal_ = reward;
            std::cout << "📈 Reward signal received: " << reward << std::endl;
        } catch (...) {
            std::cerr << "⚠️ Failed to parse reward signal" << std::endl;
        }
    }
    else if (command == "SAVE_STATE") {
        // Use the member variable save_path_ for the save location
        if (saveAgentState(save_path_)) {
            std::cout << "💾 Agent state saved to: " << save_path_ << std::endl;
        } else {
            std::cerr << "❌ Failed to save agent state" << std::endl;
        }
    }
    else {
        std::cerr << "⚠️ Unknown command: " << command << std::endl;
    }
}

void AutonomousLearningAgent::startAutonomousLearning() {
    if (is_learning_active_) return;

    is_learning_active_ = true;
    std::cout << "Starting autonomous learning mode..." << std::endl;
}

void AutonomousLearningAgent::stopAutonomousLearning() {
    if (!is_learning_active_) return;

    is_learning_active_ = false;
    std::cout << "Stopping autonomous learning mode..." << std::endl;
}

float AutonomousLearningAgent::autonomousLearningStep(float dt) {
    if (!is_learning_active_) return getLearningProgress();

    // === NLP-FOCUSED LEARNING CYCLE (No autonomous browsing) ===
    
    // Step 1: Update neural working memory with current language context
    update_working_memory();
    
    // Step 2: Process context through prefrontal cortex (language understanding)
    std::vector<float> processed_output;
    if (modules_.count("prefrontal_cortex") > 0) {
        processed_output = modules_["prefrontal_cortex"]->process(environmental_context_);
    }
    
    // Step 3: Coordinate neural modules for language processing
    coordinate_modules();
    
    // Step 4: Update language processing metrics (simplified)
    float immediate_reward = 0.1f; // Small constant reward for language learning
    
    // Step 5: Store language processing experience
    if (memory_system_) {
        storeEpisodeInMemory(immediate_reward);
    }
    
    // Step 6: Update attention weights based on language context
    update_attention_weights();

    // Log learning progress periodically
    static int step_count = 0;
    if (++step_count % 50 == 0) {
        logLearningProgress(step_count, immediate_reward);
    }

    // Return learning progress (based on accumulated experience and performance)
    return getLearningProgress();
}

void AutonomousLearningAgent::select_and_execute_action() {
    // Use decision-making system from DecisionAndActionSystems.cpp

    // Exploration vs. Exploitation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (dis(gen) < exploration_rate_) {
        // Explore: select a random action
        int random_action_idx = std::uniform_int_distribution<>(0, static_cast<int>(ActionType::BACKSPACE))(gen);
        selected_action_.type = static_cast<ActionType>(random_action_idx);
        selected_action_.confidence = 1.0f; // Confidence is high for exploration
        log_action("Exploring with random action: " + actionTypeToString(selected_action_.type));
    } else {
        // Exploit: use the decision-making system
        make_decision();
    }

    execute_action();
}

float AutonomousLearningAgent::calculate_immediate_reward() {
    float reward = 0.0f;

    // Reward for successful actions
    if (metrics_.successful_actions > 0) {
        reward += 0.1f * metrics_.successful_actions;
    }

    // Penalizing WAIT is no longer applicable

    // Reward for exploration and novelty
    float novelty_bonus = 0.0f;
    if (memory_system_ && !environmental_context_.empty()) {
        auto similar_episodes = memory_system_->retrieveSimilarEpisodes(environmental_context_, "default", 3);
        if (similar_episodes.size() < 2) {
            novelty_bonus = 0.2f; // High novelty
        } else {
            novelty_bonus = 0.05f; // Some novelty
        }
    }
    reward += novelty_bonus;

    // Reward for progressing towards a goal
    if (!learning_goals_.empty()) {
        // Implement logic to check if the agent is making progress towards its goals
        // For example, if a goal is to click a specific button, and the agent does so,
        // provide a large reward.
    }

    return std::max(-0.5f, std::min(reward, 0.5f));
}

// ============================================================================
// ADDITIONAL INTERFACE METHODS
// ============================================================================

void AutonomousLearningAgent::addLearningGoal(std::unique_ptr<AutonomousGoal> goal) {
    // Not yet implemented
}

void AutonomousLearningAgent::set_learning_goal(const std::string& goal) {
    learning_goals_.push_back(goal);
}

void AutonomousLearningAgent::execute_action(const BrowsingAction& action) {
    if (action_executor_) {
        action_executor_(action);
    }
}

void AutonomousLearningAgent::setEnvironmentSensor(std::function<BrowsingState()> sensor) {
    environment_sensor_ = sensor;
}

void AutonomousLearningAgent::setActionExecutor(std::function<void(const BrowsingAction&)> executor) {
    action_executor_ = executor;
}

bool AutonomousLearningAgent::isActionValid(const BrowsingAction& action) {
    // Basic validation, can be expanded
    if (action.type == ActionType::CLICK) {
        // Check if coordinates are within reasonable bounds
        // This requires knowledge of the screen/window size, which should be in the state
    }
    return true; // Placeholder
}

// ============================================================================
// MISSING DECLARATION IMPLEMENTATIONS (Stubs for linker resolution)
// ============================================================================

float AutonomousLearningAgent::getLearningProgress() const {
    // Simplified: derive progress from average reward proxy
    return std::clamp(metrics_.average_reward + 0.5f, 0.0f, 1.0f);
}

void AutonomousLearningAgent::storeEpisodeInMemory(float reward) {
    if (!memory_system_) return;
    std::vector<float> state_snapshot = global_state_;
    std::vector<float> action_vec(5, 0.0f);
    if (static_cast<int>(selected_action_.type) < static_cast<int>(action_vec.size())) {
        action_vec[static_cast<int>(selected_action_.type)] = 1.0f;
    }
    memory_system_->store_episode(state_snapshot, action_vec, reward, selected_action_.confidence);
}

void AutonomousLearningAgent::logLearningProgress(int step, float reward) {
    if (!detailed_logging_) return;
    std::cout << "[LearningProgress] step=" << step
              << " reward=" << reward
              << " avg_reward=" << metrics_.average_reward
              << " exploration=" << exploration_rate_ << std::endl;
}

void AutonomousLearningAgent::update_learning_goals() {
    // NLP-only mode: no dynamic goal adjustment beyond maintaining list size
    if (learning_goals_.size() > 50) {
        learning_goals_.erase(learning_goals_.begin());
    }
}

void AutonomousLearningAgent::log_action(const std::string& action) {
    if (detailed_logging_) {
        std::cout << "[ActionLog] " << action << std::endl;
    }
}

void AutonomousLearningAgent::setupDefaultLearningGoals() {
    // Not implemented in this version
}

// ============================================================================
// REAL SCREEN-BASED REINFORCEMENT LEARNING METHODS
// ============================================================================

void AutonomousLearningAgent::processRealScreenInput() {
    // This method is disabled for NLP focus - no visual processing required
    return;
}

float AutonomousLearningAgent::computeScreenBasedReward() {
    float reward = 0.0f;

    // Reward for successful actions
    if (metrics_.total_actions > 0) {
        float success_rate = static_cast<float>(metrics_.successful_actions) / metrics_.total_actions;
        reward += success_rate * 0.1f;
    }

    // Reward for exploration and discovery (simplified)
    if (exploration_rate_ > 0.5f) {
        reward += 0.02f;
    }

    // Penalty for inaction
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last_action = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_action_time_).count();
    if (time_since_last_action > 15) {
        reward -= 0.1f;
    }

    return reward;
}

// Placeholder implementations for new private methods
float AutonomousLearningAgent::evaluateGoalProgress() { return 0.0f; }
float AutonomousLearningAgent::evaluateExplorationEffectiveness() { return 0.0f; }
float AutonomousLearningAgent::evaluateActionPenalties() { return 0.0f; }
float AutonomousLearningAgent::evaluateLearningEfficiency() { return 0.0f; }
float AutonomousLearningAgent::evaluateTaskCompletion() { return 0.0f; }
float AutonomousLearningAgent::evaluateLearningImprovement() { return 0.0f; }
void AutonomousLearningAgent::updateLanguageMetrics(float comprehension_score) {}
void AutonomousLearningAgent::applyReward(float reward) {}
int AutonomousLearningAgent::getTotalNeuronCount() const { return 0; }
int AutonomousLearningAgent::getModuleNeuronCount(const std::string& module_name) const { return 0; }
std::string AutonomousLearningAgent::getCurrentTimestamp() const { return ""; }
std::vector<float> AutonomousLearningAgent::extractLanguageFeatures(const std::string& text) const { return {}; }
float AutonomousLearningAgent::computeLanguageComprehension(const std::vector<float>& neural_output) const { return 0.0f; }
std::string AutonomousLearningAgent::convertNeuralToLanguage(const std::vector<float>& neural_features) const { return ""; }
std::string AutonomousLearningAgent::generateNextWordPrediction(const std::string& context, const std::vector<float>& neural_output) { return ""; }
bool AutonomousLearningAgent::saveAgentState(const std::string& save_path) { return true; }
bool AutonomousLearningAgent::loadAgentState(const std::string& load_path) { return true; }
bool AutonomousLearningAgent::saveModule(const std::string& module_name, const std::string& save_path) { return true; }
bool AutonomousLearningAgent::loadModule(const std::string& module_name, const std::string& load_path) { return true; }
std::string AutonomousLearningAgent::getTrainingStatistics() const { return ""; }
void AutonomousLearningAgent::setTrainingStatistics(const std::string& stats_json) {}
void AutonomousLearningAgent::setPassiveMode(bool passive) { is_passive_mode_ = passive; }

// ============================================================================
// LANGUAGE TRAINING INTERFACE IMPLEMENTATION
// ============================================================================

bool AutonomousLearningAgent::processLanguageInput(const std::string& language_input) {
    try {
        std::cout << "🔤 Processing language input: " << language_input.substr(0, 50) << "..." << std::endl;

        // Convert language to neural input patterns
        std::vector<float> language_features = extractLanguageFeatures(language_input);

        // Process through language understanding modules
        if (modules_.count("prefrontal_cortex")) {
            auto language_output = modules_["prefrontal_cortex"]->process(language_features);

            // Update language understanding metrics
            float comprehension_score = computeLanguageComprehension(language_output);
            updateLanguageMetrics(comprehension_score);

            // Generate next word prediction
            std::string predicted_word = generateNextWordPrediction(language_input, language_output);

            // Output prediction in the format expected by Python script
            std::cout << "NEXT_WORD_PREDICTION:" << predicted_word << std::endl;
            std::cout.flush(); // Ensure immediate output

            return true;
        }

        return false;

    } catch (const std::exception& e) {
        std::cerr << "Failed to process language input: " << e.what() << std::endl;
        return false;
    }
}

std::string AutonomousLearningAgent::generateLanguageResponse() {
    try {
        // Generate response using motor cortex for language generation
        if (modules_.count("motor_cortex")) {
            std::vector<float> current_context = environmental_context_;
            auto response_features = modules_["motor_cortex"]->process(current_context);

            // Convert neural output to language
            return convertNeuralToLanguage(response_features);
        }

        return "I am processing your request with my neural networks.";

    } catch (const std::exception& e) {
        std::cerr << "Failed to generate language response: " << e.what() << std::endl;
        return "Error generating response.";
    }
}

void AutonomousLearningAgent::execute_action() {
    // This method is disabled for NLP focus - no actions to execute
    // Just update metrics for compatibility
    metrics_.total_actions++;
    last_action_time_ = std::chrono::steady_clock::now();
    
    if (detailed_logging_) {
        std::cout << "[NLP Agent] Action execution disabled (NLP-only mode)" << std::endl;
    }
}
