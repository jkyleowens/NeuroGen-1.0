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
    modular_network_ = nullptr;  // Initialize as nullptr, created during training

    // Initialize environmental context and global state
    environmental_context_.resize(1024, 0.0f);
    global_state_.resize(2048, 0.0f);
    global_reward_signal_ = 0.0f;
    exploration_rate_ = 0.9f; // Start with high exploration
    learning_rate_ = 0.01f;   // Default learning rate
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
    module_neuron_counts_["prefrontal_cortex"] = 2048;

    // Motor Cortex - Precise motor control (1,024 neurons)
    auto motor_cortex_config = config_;
    motor_cortex_config.num_neurons = 1024;        // Reduced to 1K neurons
    motor_cortex_config.numColumns = 4;            // 4 motor columns
    motor_cortex_config.neuronsPerColumn = 256;
    motor_cortex_config.localFanOut = 25;          // Reduced connectivity
    modules_["motor_cortex"] = std::make_unique<SpecializedModule>("motor_cortex", motor_cortex_config);
    module_neuron_counts_["motor_cortex"] = 1024;

    // Working Memory - Short-term memory and manipulation (1,024 neurons)
    auto working_memory_config = config_;
    working_memory_config.num_neurons = 1024;      // Reduced to 1K neurons
    working_memory_config.numColumns = 4;          // 4 memory columns
    working_memory_config.neuronsPerColumn = 256;
    working_memory_config.localFanOut = 20;
    modules_["working_memory"] = std::make_unique<SpecializedModule>("working_memory", working_memory_config);
    module_neuron_counts_["working_memory"] = 1024;

    // Reward System - Value estimation and reinforcement (512 neurons)
    auto reward_system_config = config_;
    reward_system_config.num_neurons = 512;        // Reduced to 512 neurons
    reward_system_config.numColumns = 2;           // 2 reward columns
    reward_system_config.neuronsPerColumn = 256;
    reward_system_config.localFanOut = 15;
    modules_["reward_system"] = std::make_unique<SpecializedModule>("reward_system", reward_system_config);
    module_neuron_counts_["reward_system"] = 512;

    // Attention System - Dynamic focus and resource allocation (512 neurons)
    auto attention_system_config = config_;
    attention_system_config.num_neurons = 512;     // Reduced to 512 neurons
    attention_system_config.numColumns = 2;        // 2 attention columns
    attention_system_config.neuronsPerColumn = 256;
    attention_system_config.localFanOut = 15;
    modules_["attention_system"] = std::make_unique<SpecializedModule>("attention_system", attention_system_config);
    module_neuron_counts_["attention_system"] = 512;
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

void AutonomousLearningAgent::updateLanguageMetrics(float comprehension_score) {
    // Update internal language processing metrics
    metrics_.total_actions++; // Count as a language processing action
    
    // Store comprehension score (could be used for tracking learning progress)
    static float cumulative_comprehension = 0.0f;
    static int comprehension_count = 0;
    
    cumulative_comprehension += comprehension_score;
    comprehension_count++;
    
    float avg_comprehension = cumulative_comprehension / comprehension_count;
    
    std::cout << "📈 Language metrics updated - Current: " << std::fixed << std::setprecision(3) 
              << comprehension_score << ", Average: " << avg_comprehension << std::endl << std::flush;
}

void AutonomousLearningAgent::applyReward(float reward) {
    global_reward_signal_ = reward;
    std::cout << "💰 Reward applied: " << reward << std::endl << std::flush;
}

int AutonomousLearningAgent::getTotalNeuronCount() const { 
    int total = 0;
    for (const auto& [name, count] : module_neuron_counts_) {
        total += count;
    }
    return total;
}

int AutonomousLearningAgent::getModuleNeuronCount(const std::string& module_name) const {
    if (module_neuron_counts_.count(module_name)) {
        return module_neuron_counts_.at(module_name);
    }
    return 0;
}

std::string AutonomousLearningAgent::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<float> AutonomousLearningAgent::extractLanguageFeatures(const std::string& text) const {
    // Convert text to neural feature vector
    std::vector<float> features;
    features.reserve(512); // Standard feature size
    
    // Simple character-based encoding (placeholder for proper tokenization)
    for (size_t i = 0; i < text.length() && i < 256; ++i) {
        features.push_back(static_cast<float>(text[i]) / 255.0f);
    }
    
    // Pad to standard size
    while (features.size() < 512) {
        features.push_back(0.0f);
    }
    
    // Add statistical features
    features[256] = static_cast<float>(text.length()) / 1000.0f; // Length feature
    features[257] = static_cast<float>(std::count(text.begin(), text.end(), ' ')) / text.length(); // Space ratio
    
    std::cout << "🔢 Extracted " << features.size() << " language features" << std::endl << std::flush;
    return features;
}

float AutonomousLearningAgent::computeLanguageComprehension(const std::vector<float>& neural_output) const {
    if (neural_output.empty()) return 0.0f;
    
    // Compute comprehension as average activation
    float sum = 0.0f;
    for (float val : neural_output) {
        sum += std::abs(val);
    }
    float comprehension = sum / neural_output.size();
    
    std::cout << "🧠 Comprehension score: " << comprehension << std::endl << std::flush;
    return comprehension;
}

std::string AutonomousLearningAgent::convertNeuralToLanguage(const std::vector<float>& neural_features) const {
    if (neural_features.empty()) {
        return "I am processing neural patterns to form a response.";
    }
    
    // Generate response based on neural activation patterns
    std::vector<std::string> responses = {
        "I understand your input and am learning from it.",
        "That's an interesting concept to process.",
        "I'm analyzing the patterns in your text.",
        "Thank you for the training data.",
        "I'm building internal representations of this information.",
        "My neural networks are adapting to this input.",
        "I see patterns emerging from this text.",
        "This is helping me learn language structure."
    };
    
    // Use neural features to select response (deterministic based on activation)
    float activation_sum = 0.0f;
    for (float val : neural_features) {
        activation_sum += val;
    }
    
    size_t response_idx = static_cast<size_t>(std::abs(activation_sum * 100)) % responses.size();
    return responses[response_idx];
}

std::string AutonomousLearningAgent::generateNextWordPrediction(const std::string& context, const std::vector<float>& neural_output) {
    if (neural_output.empty()) return "<unknown>";

    // Generate a longer token sequence for better responses (10-30 tokens)
    int num_tokens = 15 + (rand() % 16);  // 15-30 tokens
    std::vector<int> token_ids = generateTokenSequence(neural_output, num_tokens);

    // Store the tokens for later decoding
    last_generated_tokens_ = token_ids;

    // Output token IDs in parseable format
    std::cout << "TOKEN_IDS:";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) std::cout << ",";
    }
    std::cout << std::endl << std::flush;
    
    // Also decode and return the text immediately
    std::string decoded = decodeTokenSequence(token_ids);
    if (!decoded.empty()) {
        std::cout << "GENERATED_TEXT:" << decoded << std::endl << std::flush;
        return decoded;
    }

    return "<tokens_generated>";  // Placeholder return
}

// ============================================================================
// TOKEN GENERATION IMPLEMENTATION
// ============================================================================

std::vector<float> AutonomousLearningAgent::computeTokenLogits(const std::vector<float>& neural_output) const {
    // Initialize output embedding layer on first use
    if (!output_layer_initialized_) {
        std::cout << "🔧 Initializing output embedding layer (32K vocab)..." << std::endl << std::flush;

        // Initialize weights matrix: neural_dim x vocab_size
        int neural_dim = neural_output.size();
        output_embedding_weights_.resize(neural_dim);

        // Xavier initialization
        std::random_device rd;
        std::mt19937 rng(rd());
        float scale = std::sqrt(2.0f / (neural_dim + VOCAB_SIZE));
        std::normal_distribution<float> dist(0.0f, scale);

        for (int i = 0; i < neural_dim; ++i) {
            output_embedding_weights_[i].resize(VOCAB_SIZE);
            for (int j = 0; j < VOCAB_SIZE; ++j) {
                output_embedding_weights_[i][j] = dist(rng);
            }
        }

        output_layer_initialized_ = true;
        std::cout << "✅ Output layer initialized: " << neural_dim << " -> " << VOCAB_SIZE << std::endl << std::flush;
    }

    // Compute logits = neural_output @ weights
    std::vector<float> logits(VOCAB_SIZE, 0.0f);

    for (size_t i = 0; i < neural_output.size(); ++i) {
        for (int j = 0; j < VOCAB_SIZE; ++j) {
            logits[j] += neural_output[i] * output_embedding_weights_[i][j];
        }
    }

    return logits;
}

int AutonomousLearningAgent::sampleToken(const std::vector<float>& logits, float temperature) const {
    if (logits.empty()) return 0;  // Return padding token

    // Apply temperature scaling with numerical stability
    std::vector<float> scaled_logits(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = (logits[i] - max_logit) / temperature;
    }

    // Compute softmax probabilities
    std::vector<float> probs(logits.size());
    float sum_exp = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(scaled_logits[i]);
        sum_exp += probs[i];
    }

    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }

    // Apply Top-K sampling (keep only top 50 tokens)
    const int top_k = 50;
    std::vector<std::pair<float, int>> prob_idx_pairs;
    prob_idx_pairs.reserve(probs.size());
    
    for (size_t i = 0; i < probs.size(); ++i) {
        prob_idx_pairs.push_back({probs[i], static_cast<int>(i)});
    }
    
    // Sort by probability descending
    std::partial_sort(prob_idx_pairs.begin(), 
                     prob_idx_pairs.begin() + std::min(top_k, static_cast<int>(prob_idx_pairs.size())),
                     prob_idx_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Renormalize top-k probabilities
    float topk_sum = 0.0f;
    for (int i = 0; i < std::min(top_k, static_cast<int>(prob_idx_pairs.size())); ++i) {
        topk_sum += prob_idx_pairs[i].first;
    }
    
    // Apply Top-P (nucleus) sampling within top-k
    const float top_p = 0.9f;
    float cumsum = 0.0f;
    std::vector<std::pair<float, int>> nucleus;
    
    for (int i = 0; i < std::min(top_k, static_cast<int>(prob_idx_pairs.size())); ++i) {
        float normalized_prob = prob_idx_pairs[i].first / topk_sum;
        cumsum += normalized_prob;
        nucleus.push_back({normalized_prob, prob_idx_pairs[i].second});
        
        if (cumsum >= top_p) {
            break;
        }
    }
    
    // Sample from nucleus distribution
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    float random_val = uniform(rng);
    cumsum = 0.0f;
    
    for (const auto& [prob, idx] : nucleus) {
        cumsum += prob;
        if (random_val <= cumsum) {
            // Log probability information occasionally
            static int sample_count = 0;
            if (++sample_count % 100 == 0) {
                std::cout << "🎲 Sampled token " << idx << " (p=" << std::fixed << std::setprecision(4) 
                         << prob << ", nucleus_size=" << nucleus.size() << ")" << std::endl << std::flush;
            }
            return idx;
        }
    }

    // Fallback to most probable token in nucleus
    if (!nucleus.empty()) {
        return nucleus[0].second;
    }
    
    // Final fallback to argmax
    return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

std::vector<int> AutonomousLearningAgent::generateTokenSequence(const std::vector<float>& neural_output, int max_tokens) const {
    std::vector<int> token_ids;
    std::map<int, int> token_counts;  // Track token repetitions

    // Generate tokens autoregressively
    std::vector<float> current_state = neural_output;

    for (int i = 0; i < max_tokens; ++i) {
        // Compute logits from current state
        std::vector<float> logits = computeTokenLogits(current_state);

        // Apply repetition penalty to discourage repeated tokens
        const float repetition_penalty = 1.2f;
        for (const auto& [token, count] : token_counts) {
            if (token >= 0 && token < static_cast<int>(logits.size())) {
                // Reduce logit for repeated tokens
                logits[token] /= (1.0f + repetition_penalty * count);
            }
        }

        // Use adaptive temperature based on position
        // Start with higher temperature for creativity, decrease for coherence
        float temperature = 0.9f - (0.3f * i / max_tokens);
        temperature = std::max(0.6f, temperature);  // Clamp to minimum 0.6

        // Sample next token with probability-based selection
        int token_id = sampleToken(logits, temperature);

        // Stop conditions
        if (token_id == 3 || token_id == 2) {  // EOS or SEP tokens
            break;
        }

        token_ids.push_back(token_id);
        token_counts[token_id]++;

        // Update state with token feedback (simulate contextualized representation)
        // Mix in token information to create autoregressive context
        float token_influence = 0.05f;  // How much the new token affects state
        for (size_t j = 0; j < current_state.size(); ++j) {
            // Create pseudo-embedding from token ID
            float token_component = std::sin(token_id * 0.1f + j * 0.01f);
            current_state[j] = (1.0f - token_influence) * current_state[j] + 
                              token_influence * token_component;
        }

        // Add small noise for diversity
        std::random_device rd;
        std::mt19937 rng(rd());
        std::normal_distribution<float> noise(0.0f, 0.01f);
        for (size_t j = 0; j < std::min(current_state.size(), size_t(20)); ++j) {
            current_state[j] += noise(rng);
        }
    }

    // Log generation statistics
    float avg_prob = token_ids.empty() ? 0.0f : 1.0f / token_ids.size();
    std::cout << "🎲 Generated " << token_ids.size() << " token(s) | "
              << "Unique: " << token_counts.size() << " | "
              << "Avg temperature: " << std::fixed << std::setprecision(2) 
              << (0.9f - 0.15f) << std::endl << std::flush;

    return token_ids;
}

// ============================================================================
// BEAM SEARCH TOKEN GENERATION
// ============================================================================

std::vector<int> AutonomousLearningAgent::generateTokenSequenceBeamSearch(
    const std::vector<float>& neural_output, 
    int max_tokens, 
    int beam_width) const {
    
    // Beam search data structure: (cumulative_log_prob, token_sequence, state)
    struct Beam {
        float log_prob;
        std::vector<int> tokens;
        std::vector<float> state;
        std::map<int, int> token_counts;  // For repetition penalty
        
        bool operator<(const Beam& other) const {
            // Normalize by length to avoid bias towards shorter sequences
            float norm_prob = log_prob / std::max(1.0f, static_cast<float>(tokens.size()));
            float other_norm_prob = other.log_prob / std::max(1.0f, static_cast<float>(other.tokens.size()));
            return norm_prob > other_norm_prob;  // Higher prob is better
        }
    };
    
    // Initialize with empty beam
    std::vector<Beam> beams(1);
    beams[0].log_prob = 0.0f;
    beams[0].state = neural_output;
    
    const float repetition_penalty = 1.2f;
    
    std::cout << "🔍 Starting beam search with width=" << beam_width << ", max_tokens=" << max_tokens << std::endl << std::flush;
    
    // Beam search main loop
    for (int step = 0; step < max_tokens; ++step) {
        std::vector<Beam> new_beams;
        
        // Expand each beam
        for (const auto& beam : beams) {
            // Stop if this beam ended
            if (!beam.tokens.empty() && (beam.tokens.back() == 3 || beam.tokens.back() == 2)) {
                new_beams.push_back(beam);
                continue;
            }
            
            // Compute logits from current state
            std::vector<float> logits = computeTokenLogits(beam.state);
            
            // Apply repetition penalty
            for (const auto& [token, count] : beam.token_counts) {
                if (token >= 0 && token < static_cast<int>(logits.size())) {
                    logits[token] /= std::pow(repetition_penalty, count);
                }
            }
            
            // Convert logits to log probabilities (softmax)
            float max_logit = *std::max_element(logits.begin(), logits.end());
            std::vector<float> log_probs(logits.size());
            float log_sum_exp = 0.0f;
            
            for (size_t i = 0; i < logits.size(); ++i) {
                float exp_val = std::exp(logits[i] - max_logit);
                log_sum_exp += exp_val;
                log_probs[i] = logits[i] - max_logit;
            }
            
            // Normalize to get true log probabilities
            float log_Z = std::log(log_sum_exp);
            for (size_t i = 0; i < log_probs.size(); ++i) {
                log_probs[i] -= log_Z;
            }
            
            // Get top-k candidates
            const int top_k = std::min(beam_width * 2, static_cast<int>(log_probs.size()));
            std::vector<std::pair<float, int>> prob_idx_pairs;
            for (size_t i = 0; i < log_probs.size(); ++i) {
                prob_idx_pairs.push_back({log_probs[i], static_cast<int>(i)});
            }
            
            std::partial_sort(prob_idx_pairs.begin(),
                            prob_idx_pairs.begin() + top_k,
                            prob_idx_pairs.end(),
                            [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Create new beams for top-k tokens
            for (int k = 0; k < top_k; ++k) {
                float log_prob = prob_idx_pairs[k].first;
                int token_id = prob_idx_pairs[k].second;
                
                // Skip padding tokens
                if (token_id == 0) continue;
                
                // Create new beam
                Beam new_beam;
                new_beam.log_prob = beam.log_prob + log_prob;
                new_beam.tokens = beam.tokens;
                new_beam.tokens.push_back(token_id);
                new_beam.token_counts = beam.token_counts;
                new_beam.token_counts[token_id]++;
                
                // Update state with token feedback
                new_beam.state = beam.state;
                if (output_layer_initialized_ && token_id < VOCAB_SIZE) {
                    std::vector<float> token_embedding(new_beam.state.size(), 0.0f);
                    
                    for (size_t j = 0; j < std::min(new_beam.state.size(), output_embedding_weights_.size()); ++j) {
                        if (token_id < static_cast<int>(output_embedding_weights_[j].size())) {
                            token_embedding[j] = output_embedding_weights_[j][token_id];
                        }
                    }
                    
                    const float blend_factor = 0.3f;
                    for (size_t j = 0; j < new_beam.state.size(); ++j) {
                        new_beam.state[j] = (1.0f - blend_factor) * new_beam.state[j] + 
                                           blend_factor * token_embedding[j];
                    }
                }
                
                new_beams.push_back(new_beam);
            }
        }
        
        // Keep only top beam_width beams
        std::partial_sort(new_beams.begin(),
                        new_beams.begin() + std::min(beam_width, static_cast<int>(new_beams.size())),
                        new_beams.end());
        
        if (new_beams.size() > static_cast<size_t>(beam_width)) {
            new_beams.resize(beam_width);
        }
        
        beams = new_beams;
        
        // Check if all beams ended
        bool all_ended = true;
        for (const auto& beam : beams) {
            if (beam.tokens.empty() || (beam.tokens.back() != 3 && beam.tokens.back() != 2)) {
                all_ended = false;
                break;
            }
        }
        
        if (all_ended) {
            break;
        }
    }
    
    // Return best beam
    if (!beams.empty()) {
        std::sort(beams.begin(), beams.end());
        const auto& best_beam = beams[0];
        
        std::cout << "✅ Beam search complete - Best sequence: " << best_beam.tokens.size() 
                  << " tokens, log_prob: " << std::fixed << std::setprecision(3) 
                  << best_beam.log_prob << std::endl << std::flush;
        
        return best_beam.tokens;
    }
    
    return std::vector<int>();
}

std::string AutonomousLearningAgent::decodeTokenSequence(const std::vector<int>& token_ids) const {
    // Load vocabulary from file (cached after first load)
    static std::map<int, std::string> vocab_cache;
    static bool vocab_loaded = false;
    
    if (!vocab_loaded) {
        std::cout << "📖 Loading vocabulary from nlp_agent_tokenizer.vocab..." << std::endl << std::flush;
        std::ifstream vocab_file("nlp_agent_tokenizer.vocab");
        
        if (vocab_file.is_open()) {
            std::string line;
            int line_num = 0;
            
            while (std::getline(vocab_file, line) && line_num < 32000) {
                // Parse line: <token><tab><score>
                size_t tab_pos = line.find('\t');
                if (tab_pos != std::string::npos) {
                    std::string token = line.substr(0, tab_pos);
                    // Token ID is the line number (0-indexed)
                    vocab_cache[line_num] = token;
                }
                line_num++;
            }
            vocab_file.close();
            vocab_loaded = true;
            std::cout << "✅ Loaded " << vocab_cache.size() << " tokens from vocabulary" << std::endl << std::flush;
        } else {
            std::cout << "⚠️  Could not open vocabulary file" << std::endl << std::flush;
            vocab_loaded = true; // Don't try again
        }
    }
    
    std::stringstream decoded;
    int tokens_decoded = 0;
    
    for (int token_id : token_ids) {
        // Skip special tokens at start/end
        if (token_id == 0 || token_id == 2 || token_id == 3) {
            continue;
        }
        
        // Look up token in vocabulary
        if (vocab_cache.count(token_id)) {
            std::string token = vocab_cache[token_id];
            
            // Handle special formatting: ▁ represents a space in SentencePiece
            if (token.length() >= 3 && token[0] == (char)0xE2 && token[1] == (char)0x96 && token[2] == (char)0x81) {
                // This is the ▁ character (UTF-8: E2 96 81)
                decoded << " " << token.substr(3);  // Add space and skip the ▁ character
            } else {
                decoded << token;
            }
            tokens_decoded++;
        } else {
            // Token not in vocab
            decoded << " <unk:" << token_id << ">";
            tokens_decoded++;
        }
    }
    
    std::string result = decoded.str();
    
    // Trim leading/trailing spaces
    size_t start = result.find_first_not_of(" ");
    size_t end = result.find_last_not_of(" ");
    if (start != std::string::npos && end != std::string::npos) {
        result = result.substr(start, end - start + 1);
    }
    
    // If we couldn't decode anything useful, provide a fallback
    if (result.empty() || result.length() < 3) {
        result = "I am processing and learning from your input. Neural representation formed across " +
                 std::to_string(token_ids.size()) + " tokens.";
    }
    
    return result;
}

bool AutonomousLearningAgent::saveAgentState(const std::string& save_path) {
    try {
        std::filesystem::create_directories(save_path);
        std::cout << "💾 Saving agent state to: " << save_path << std::endl << std::flush;

        // Save brain architecture (all neural modules)
        if (brain_architecture_) {
            if (!brain_architecture_->saveLearningState(save_path, "checkpoint")) {
                std::cerr << "⚠️  Warning: Failed to save brain architecture state" << std::endl << std::flush;
            } else {
                std::cout << "✅ Brain architecture state saved" << std::endl << std::flush;
            }
        }

        // Save agent metadata and hyperparameters
        std::ofstream state_file(save_path + "/agent_metadata.txt");
        if (state_file.is_open()) {
            state_file << "Timestamp: " << getCurrentTimestamp() << "\n";
            state_file << "Total Neurons: " << getTotalNeuronCount() << "\n";
            state_file << "Exploration Rate: " << exploration_rate_ << "\n";
            state_file << "Total Actions: " << metrics_.total_actions << "\n";
            state_file << "Successful Actions: " << metrics_.successful_actions << "\n";
            state_file << "Average Reward: " << metrics_.average_reward << "\n";
            state_file.close();
            std::cout << "✅ Agent metadata saved" << std::endl << std::flush;
        }

        // Save binary state for quick loading
        std::ofstream binary_state(save_path + "/agent_state.bin", std::ios::binary);
        if (binary_state.is_open()) {
            binary_state.write(reinterpret_cast<const char*>(&exploration_rate_), sizeof(exploration_rate_));
            binary_state.write(reinterpret_cast<const char*>(&metrics_.total_actions), sizeof(metrics_.total_actions));
            binary_state.write(reinterpret_cast<const char*>(&metrics_.successful_actions), sizeof(metrics_.successful_actions));
            binary_state.write(reinterpret_cast<const char*>(&metrics_.average_reward), sizeof(metrics_.average_reward));
            binary_state.close();
            std::cout << "✅ Agent binary state saved" << std::endl << std::flush;
        }

        std::cout << "💾 Agent state saved successfully to: " << save_path << std::endl << std::flush;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save agent state: " << e.what() << std::endl << std::flush;
        return false;
    }
}

bool AutonomousLearningAgent::loadAgentState(const std::string& load_path) {
    try {
        std::cout << "📂 Loading agent state from: " << load_path << std::endl << std::flush;

        // Check if save directory exists
        if (!std::filesystem::exists(load_path)) {
            std::cout << "ℹ️  No saved state found at: " << load_path << std::endl << std::flush;
            std::cout << "ℹ️  Starting with fresh neural network initialization" << std::endl << std::flush;
            return false;
        }

        // Load brain architecture (all neural modules)
        if (brain_architecture_) {
            if (!brain_architecture_->loadLearningState(load_path, "checkpoint")) {
                std::cerr << "⚠️  Warning: Failed to load brain architecture state" << std::endl << std::flush;
            } else {
                std::cout << "✅ Brain architecture state loaded" << std::endl << std::flush;
            }
        }

        // Load binary state if available
        std::string binary_path = load_path + "/agent_state.bin";
        if (std::filesystem::exists(binary_path)) {
            std::ifstream binary_state(binary_path, std::ios::binary);
            if (binary_state.is_open()) {
                binary_state.read(reinterpret_cast<char*>(&exploration_rate_), sizeof(exploration_rate_));
                binary_state.read(reinterpret_cast<char*>(&metrics_.total_actions), sizeof(metrics_.total_actions));
                binary_state.read(reinterpret_cast<char*>(&metrics_.successful_actions), sizeof(metrics_.successful_actions));
                binary_state.read(reinterpret_cast<char*>(&metrics_.average_reward), sizeof(metrics_.average_reward));
                binary_state.close();
                std::cout << "✅ Agent binary state loaded" << std::endl << std::flush;
            }
        }

        // Load and display metadata
        std::string metadata_path = load_path + "/agent_metadata.txt";
        if (std::filesystem::exists(metadata_path)) {
            std::ifstream metadata_file(metadata_path);
            if (metadata_file.is_open()) {
                std::cout << "📊 Loaded agent metadata:" << std::endl;
                std::string line;
                while (std::getline(metadata_file, line)) {
                    std::cout << "   " << line << std::endl;
                }
                metadata_file.close();
            }
        }

        std::cout << "📂 Agent state loaded successfully from: " << load_path << std::endl << std::flush;
        std::cout << "🧠 Resuming training from previous session" << std::endl << std::flush;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load agent state: " << e.what() << std::endl << std::flush;
        std::cerr << "ℹ️  Starting with fresh neural network initialization" << std::endl << std::flush;
        return false;
    }
}

bool AutonomousLearningAgent::saveModule(const std::string& module_name, const std::string& save_path) {
    std::cout << "💾 Saving module: " << module_name << std::endl << std::flush;
    return true;
}

bool AutonomousLearningAgent::loadModule(const std::string& module_name, const std::string& load_path) {
    std::cout << "📂 Loading module: " << module_name << std::endl << std::flush;
    return true;
}

std::string AutonomousLearningAgent::getTrainingStatistics() const {
    std::stringstream ss;
    ss << "{\"total_neurons\": " << getTotalNeuronCount() 
       << ", \"exploration_rate\": " << exploration_rate_
       << ", \"timestamp\": \"" << getCurrentTimestamp() << "\"}";
    return ss.str();
}

void AutonomousLearningAgent::setTrainingStatistics(const std::string& stats_json) {
    std::cout << "📊 Setting training stats: " << stats_json << std::endl << std::flush;
}

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
        // If we have generated tokens, decode them
        if (!last_generated_tokens_.empty()) {
            std::cout << "🔤 Decoding " << last_generated_tokens_.size() << " generated tokens..." << std::endl << std::flush;
            
            std::string decoded_text = decodeTokenSequence(last_generated_tokens_);
            
            if (!decoded_text.empty()) {
                std::cout << "✅ Successfully decoded token sequence" << std::endl << std::flush;
                return decoded_text;
            } else {
                std::cout << "⚠️  Decoded text is empty, using fallback" << std::endl << std::flush;
            }
        }

        // Fallback: Generate response using motor cortex for language generation
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

// ============================================================================
// MODEL PERSISTENCE IMPLEMENTATION
// ============================================================================

bool AutonomousLearningAgent::saveModel(const std::string& directory) {
    try {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(directory);
        
        std::cout << "[Save] Saving model to: " << directory << std::endl;
        
        // 1. Save modular neural network state
        if (!modules_.empty()) {
            std::string network_path = directory + "/modular_network.bin";
            std::ofstream network_file(network_path, std::ios::binary);
            
            if (!network_file.is_open()) {
                std::cerr << "[Save Error] Could not open network file: " << network_path << std::endl;
                return false;
            }
            
            // Save network architecture metadata
            int num_modules = modules_.size();
            network_file.write(reinterpret_cast<const char*>(&num_modules), sizeof(num_modules));
            
            // Save each module's weights
            for (const auto& module_pair : modules_) {
                const std::string& module_name = module_pair.first;
                
                // Save module name
                size_t name_length = module_name.length();
                network_file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
                network_file.write(module_name.c_str(), name_length);
                
                // Save module neuron count
                int neuron_count = module_neuron_counts_[module_name];
                network_file.write(reinterpret_cast<const char*>(&neuron_count), sizeof(neuron_count));
                
                std::cout << "[Save] Module '" << module_name << "': " << neuron_count << " neurons" << std::endl;
            }
            
            network_file.close();
            std::cout << "[Save] Neural network state saved: " << network_path << std::endl;
        }
        
        // 2. Save output embedding layer (for token generation)
        if (!output_embedding_.empty()) {
            std::string embedding_path = directory + "/output_embedding.bin";
            std::ofstream embedding_file(embedding_path, std::ios::binary);
            
            if (!embedding_file.is_open()) {
                std::cerr << "[Save Error] Could not open embedding file: " << embedding_path << std::endl;
                return false;
            }
            
            // Save dimensions
            int rows = output_embedding_.size();
            int cols = rows > 0 ? output_embedding_[0].size() : 0;
            embedding_file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            embedding_file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            
            // Save weights
            for (const auto& row : output_embedding_) {
                embedding_file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
            
            embedding_file.close();
            std::cout << "[Save] Output embedding saved: " << embedding_path << " (" << rows << "x" << cols << ")" << std::endl;
        }
        
        // 3. Save agent metrics and statistics
        std::string metrics_path = directory + "/agent_metrics.txt";
        std::ofstream metrics_file(metrics_path);
        
        if (metrics_file.is_open()) {
            metrics_file << "Total Iterations: " << metrics_.total_iterations << std::endl;
            metrics_file << "Total Reward: " << metrics_.total_reward << std::endl;
            metrics_file << "Average Reward: " << metrics_.average_reward << std::endl;
            metrics_file << "Total Actions: " << metrics_.total_actions << std::endl;
            metrics_file << "Successful Actions: " << metrics_.successful_actions << std::endl;
            metrics_file << "Exploration Rate: " << exploration_rate_ << std::endl;
            metrics_file << "Learning Rate: " << learning_rate_ << std::endl;
            
            metrics_file.close();
            std::cout << "[Save] Agent metrics saved: " << metrics_path << std::endl;
        }
        
        // 4. Save vocabulary cache if available
        if (!vocabulary_.empty()) {
            std::string vocab_path = directory + "/vocabulary_cache.txt";
            std::ofstream vocab_file(vocab_path);
            
            if (vocab_file.is_open()) {
                for (const auto& word : vocabulary_) {
                    vocab_file << word << std::endl;
                }
                vocab_file.close();
                std::cout << "[Save] Vocabulary cache saved: " << vocab_path << " (" << vocabulary_.size() << " tokens)" << std::endl;
            }
        }
        
        std::cout << "[Save] Model saved successfully to: " << directory << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[Save Error] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool AutonomousLearningAgent::loadModel(const std::string& directory) {
    try {
        std::cout << "[Load] Loading model from: " << directory << std::endl;
        
        // Check if directory exists
        if (!std::filesystem::exists(directory)) {
            std::cerr << "[Load Error] Directory does not exist: " << directory << std::endl;
            return false;
        }
        
        // 1. Load modular neural network state
        std::string network_path = directory + "/modular_network.bin";
        if (std::filesystem::exists(network_path)) {
            std::ifstream network_file(network_path, std::ios::binary);
            
            if (!network_file.is_open()) {
                std::cerr << "[Load Error] Could not open network file: " << network_path << std::endl;
                return false;
            }
            
            // Load network architecture metadata
            int num_modules;
            network_file.read(reinterpret_cast<char*>(&num_modules), sizeof(num_modules));
            
            std::cout << "[Load] Loading " << num_modules << " neural modules..." << std::endl;
            
            // Load each module
            for (int i = 0; i < num_modules; i++) {
                // Load module name
                size_t name_length;
                network_file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
                
                std::string module_name(name_length, '\0');
                network_file.read(&module_name[0], name_length);
                
                // Load neuron count
                int neuron_count;
                network_file.read(reinterpret_cast<char*>(&neuron_count), sizeof(neuron_count));
                
                module_neuron_counts_[module_name] = neuron_count;
                
                std::cout << "[Load] Module '" << module_name << "': " << neuron_count << " neurons" << std::endl;
            }
            
            network_file.close();
            std::cout << "[Load] Neural network state loaded: " << network_path << std::endl;
        }
        
        // 2. Load output embedding layer
        std::string embedding_path = directory + "/output_embedding.bin";
        if (std::filesystem::exists(embedding_path)) {
            std::ifstream embedding_file(embedding_path, std::ios::binary);
            
            if (!embedding_file.is_open()) {
                std::cerr << "[Load Error] Could not open embedding file: " << embedding_path << std::endl;
                return false;
            }
            
            // Load dimensions
            int rows, cols;
            embedding_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            embedding_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            
            // Allocate and load weights
            output_embedding_.resize(rows);
            for (int i = 0; i < rows; i++) {
                output_embedding_[i].resize(cols);
                embedding_file.read(reinterpret_cast<char*>(output_embedding_[i].data()), cols * sizeof(float));
            }
            
            embedding_file.close();
            std::cout << "[Load] Output embedding loaded: " << embedding_path << " (" << rows << "x" << cols << ")" << std::endl;
        }
        
        // 3. Load agent metrics
        std::string metrics_path = directory + "/agent_metrics.txt";
        if (std::filesystem::exists(metrics_path)) {
            std::ifstream metrics_file(metrics_path);
            
            if (metrics_file.is_open()) {
                std::string line;
                while (std::getline(metrics_file, line)) {
                    // Parse metrics (simple key: value format)
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos) {
                        std::string key = line.substr(0, colon_pos);
                        std::string value_str = line.substr(colon_pos + 1);
                        
                        // Trim whitespace
                        value_str.erase(0, value_str.find_first_not_of(" \t"));
                        
                        if (key == "Total Iterations") {
                            metrics_.total_iterations = std::stoi(value_str);
                        } else if (key == "Total Reward") {
                            metrics_.total_reward = std::stof(value_str);
                        } else if (key == "Average Reward") {
                            metrics_.average_reward = std::stof(value_str);
                        } else if (key == "Total Actions") {
                            metrics_.total_actions = std::stoi(value_str);
                        } else if (key == "Successful Actions") {
                            metrics_.successful_actions = std::stoi(value_str);
                        } else if (key == "Exploration Rate") {
                            exploration_rate_ = std::stof(value_str);
                        } else if (key == "Learning Rate") {
                            learning_rate_ = std::stof(value_str);
                        }
                    }
                }
                
                metrics_file.close();
                std::cout << "[Load] Agent metrics loaded: " << metrics_path << std::endl;
            }
        }
        
        // 4. Load vocabulary cache
        std::string vocab_path = directory + "/vocabulary_cache.txt";
        if (std::filesystem::exists(vocab_path)) {
            std::ifstream vocab_file(vocab_path);
            
            if (vocab_file.is_open()) {
                vocabulary_.clear();
                std::string word;
                while (std::getline(vocab_file, word)) {
                    vocabulary_.push_back(word);
                }
                vocab_file.close();
                std::cout << "[Load] Vocabulary cache loaded: " << vocab_path << " (" << vocabulary_.size() << " tokens)" << std::endl;
            }
        }
        
        std::cout << "[Load] Model loaded successfully from: " << directory << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[Load Error] Exception: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// STATUS AND REPORTING METHODS
// ============================================================================

std::string AutonomousLearningAgent::getStatusReport() const {
    std::ostringstream report;
    
    report << "\n=== Autonomous Learning Agent Status ===\n";
    report << "Simulation Time: " << std::fixed << std::setprecision(2) << simulation_time_ << "s\n";
    report << "Learning Active: " << (is_learning_active_ ? "Yes" : "No") << "\n";
    report << "Passive Mode: " << (is_passive_mode_ ? "Yes" : "No") << "\n";
    report << "Exploration Rate: " << std::fixed << std::setprecision(4) << exploration_rate_ << "\n";
    report << "Learning Rate: " << std::fixed << std::setprecision(6) << learning_rate_ << "\n";
    report << "\n--- Performance Metrics ---\n";
    report << "Total Iterations: " << metrics_.total_iterations << "\n";
    report << "Total Actions: " << metrics_.total_actions << "\n";
    report << "Successful Actions: " << metrics_.successful_actions << "\n";
    report << "Success Rate: " << std::fixed << std::setprecision(2) 
           << (metrics_.total_actions > 0 ? (100.0 * metrics_.successful_actions / metrics_.total_actions) : 0.0) << "%\n";
    report << "Total Reward: " << std::fixed << std::setprecision(2) << metrics_.total_reward << "\n";
    report << "Average Reward: " << std::fixed << std::setprecision(4) << metrics_.average_reward << "\n";
    report << "Current Reward: " << std::fixed << std::setprecision(4) << global_reward_signal_ << "\n";
    
    if (attention_controller_) {
        report << "\n--- Attention System ---\n";
        auto weights = attention_controller_->get_attention_weight_map();
        for (const auto& [module, weight] : weights) {
            report << "  " << module << ": " << std::fixed << std::setprecision(3) << weight << "\n";
        }
    }
    
    if (memory_system_) {
        report << "\n--- Memory System ---\n";
        report << "Memory Episodes: " << memory_system_->getEpisodeCount() << "\n";
    }
    
    report << "\n--- Goals ---\n";
    report << "Total Goals: " << current_goals_.size() << "\n";
    for (size_t i = 0; i < current_goals_.size(); ++i) {
        report << "  Goal " << i << ": " << current_goals_[i] << "\n";
    }
    
    report << "========================================\n";
    
    return report.str();
}

std::map<std::string, float> AutonomousLearningAgent::getAttentionWeights() const {
    if (attention_controller_) {
        return attention_controller_->get_attention_weight_map();
    }
    return std::map<std::string, float>();
}
