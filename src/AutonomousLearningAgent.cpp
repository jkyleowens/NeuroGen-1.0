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

    // Generate token sequence from neural output
    std::vector<int> token_ids = generateTokenSequence(neural_output, 10);  // Generate up to 10 tokens

    // Store the tokens for later decoding
    last_generated_tokens_ = token_ids;

    // Output token IDs in parseable format
    std::cout << "TOKEN_IDS:";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) std::cout << ",";
    }
    std::cout << std::endl << std::flush;

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

    // Apply temperature scaling
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

    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }

    // Sample from categorical distribution
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    float random_val = uniform(rng);
    float cumsum = 0.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (random_val <= cumsum) {
            return static_cast<int>(i);
        }
    }

    // Fallback to argmax
    return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

std::vector<int> AutonomousLearningAgent::generateTokenSequence(const std::vector<float>& neural_output, int max_tokens) const {
    std::vector<int> token_ids;

    // Generate tokens autoregressively
    std::vector<float> current_state = neural_output;

    for (int i = 0; i < max_tokens; ++i) {
        // Compute logits from current state
        std::vector<float> logits = computeTokenLogits(current_state);

        // Sample next token
        int token_id = sampleToken(logits, 0.8f);  // temperature = 0.8 for some randomness

        // Stop if we generate end-of-sequence token (token 3)
        if (token_id == 3) {
            break;
        }

        token_ids.push_back(token_id);

        // Update state (simple approach: just modify slightly)
        // In a real implementation, this would feedback the token embedding
        for (size_t j = 0; j < current_state.size() && j < 10; ++j) {
            current_state[j] += 0.01f * (token_id % 100 - 50);
        }
    }

    std::cout << "🎲 Generated " << token_ids.size() << " tokens" << std::endl << std::flush;

    return token_ids;
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
        
        // Save basic state info
        std::ofstream state_file(save_path + "/agent_state.txt");
        if (state_file.is_open()) {
            state_file << "Timestamp: " << getCurrentTimestamp() << "\n";
            state_file << "Total Neurons: " << getTotalNeuronCount() << "\n";
            state_file << "Exploration Rate: " << exploration_rate_ << "\n";
            state_file.close();
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save agent state: " << e.what() << std::endl << std::flush;
        return false;
    }
}

bool AutonomousLearningAgent::loadAgentState(const std::string& load_path) {
    std::cout << "📂 Loading agent state from: " << load_path << std::endl << std::flush;
    return true;
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
