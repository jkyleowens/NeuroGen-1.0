// ============================================================================
// AUTONOMOUS LEARNING AGENT HEADER
// File: include/NeuroGen/AutonomousLearningAgent.h
// ============================================================================

#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/SpecializedModule.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/InputController.h"
#include "NeuroGen/SafetyManager.h"
#include "NeuroGen/Action.h"
#include "NeuroGen/MemorySystem.h"
#include "NeuroGen/AttentionController.h"
#include "ModularNeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <map>
#include <filesystem>

/**
 * @brief Defines the operating modes for the agent.
 */
enum class OperatingMode {
    IDLE,
    AUTONOMOUS,
    MANUAL_CONTROL
};

/**
 * @brief Autonomous goal structure for learning objectives
 */
struct AutonomousGoal {
    std::string goal_id;
    std::string description;
    float priority;
    std::vector<std::string> success_criteria;
    bool is_active;
    
    AutonomousGoal() : priority(0.5f), is_active(false) {}
};

// ============================================================================
// CORE STRUCTURES
// ============================================================================

/**
 * @brief Browsing state representation
 */
struct BrowsingState {
    std::string current_url;
    std::vector<std::string> page_elements;
    int scroll_position;
    int window_width;
    int window_height;
    bool page_loading;
    
    BrowsingState() : scroll_position(0), window_width(1920), window_height(1080), page_loading(false) {}
};

// Using BrowsingAction from Action.h

// ============================================================================
// AUTONOMOUS LEARNING AGENT CLASS DECLARATION
// ============================================================================

/**
 * @brief Autonomous Learning Agent for Complex Decision Making
 * 
 * This class implements an autonomous agent capable of:
 * - Multi-modal perception and processing
 * - Dynamic decision making and action execution  
 * - Continuous learning from environmental feedback
 * - Modular neural network coordination
 * - Adaptive exploration and exploitation strategies
 */
class AutonomousLearningAgent {
public:
    // Constructor and Destructor
    AutonomousLearningAgent(const NetworkConfig& config);
    ~AutonomousLearningAgent();

    // Core Lifecycle Methods
    bool initialize(bool reset_model = false);
    void update(float dt);
    void shutdown();

    // Autonomous Learning Control
    void startAutonomousLearning();
    void stopAutonomousLearning();

    // Command Handling
    void handleCommand(const std::string& command);

    // State Management
    bool saveAgentState(const std::string& save_path);
    bool loadAgentState(const std::string& save_path);

    // Configuration
    void setPassiveMode(bool passive);
    bool isPassiveMode() const;

    // Detailed logging configuration
    void setDetailedLogging(bool detailed_logging) { detailed_logging_ = detailed_logging; }

    // Status query methods
    bool isLearningActive() const { return is_learning_active_; }
    OperatingMode getCurrentMode() const { return current_mode_; }
    const BrowsingAction& getCurrentAction() const { return selected_action_; }
    float getDecisionConfidence() const { return selected_action_.confidence; }
    std::string getCurrentDecision() const { 
        switch(selected_action_.type) {
            case ActionType::CLICK: 
                return "Click at (" + std::to_string(selected_action_.x_coordinate) + 
                       ", " + std::to_string(selected_action_.y_coordinate) + ")";
            case ActionType::SCROLL: 
                return std::string("Scroll ") + 
                       (selected_action_.scroll_direction == ScrollDirection::UP ? "up" : "down") + 
                       " by " + std::to_string(selected_action_.scroll_amount);
            case ActionType::TYPE: 
                return "Type: " + selected_action_.text_content;
            case ActionType::ENTER: 
                return "Press Enter";
            case ActionType::BACKSPACE: 
                return "Press Backspace";
            case ActionType::NAVIGATE: 
                return "Navigate to: " + selected_action_.url;
            case ActionType::WAIT: 
                return "Wait for " + std::to_string(selected_action_.wait_duration_ms) + "ms";
            case ActionType::NONE:
            default:
                return "No action";
        }
    }

    // ========================================================================
    // AUTONOMOUS LEARNING INTERFACE
    // ========================================================================

    /**
     * @brief Perform one step of autonomous learning
     * @param dt Time step
     * @return Learning progress indicator [0-1]
     */
    float autonomousLearningStep(float dt);
    
    /**
     * @brief Add a learning goal for the agent
     * @param goal Autonomous goal to pursue
     */
    void addLearningGoal(std::unique_ptr<AutonomousGoal> goal);

    // Status and Metrics
    std::string getStatusReport() const;
    float getLearningProgress() const;
    std::map<std::string, float> getAttentionWeights() const;
    BrowsingState getCurrentEnvironmentState() const;
    std::string getTrainingStatistics() const;
    void setTrainingStatistics(const std::string& stats_json);
    bool processLanguageInput(const std::string& language_input);
    void applyReward(float reward);
    std::string generateLanguageResponse();
    void updateLanguageMetrics(float comprehension_score);
    std::string generateNextWordPrediction(const std::string& context, const std::vector<float>& neural_output);

    // Streaming token generation interface
    std::vector<float> getCurrentNeuralOutput() const;
    int generateNextToken(std::vector<float>& current_state, float temperature = 0.8f);
    std::string decodeToken(int token_id) const;
    bool isEndOfSequenceToken(int token_id) const { return token_id == 3; }

    // ========================================================================
    // CORE PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process visual input from environment (NLP mode: disabled)
     */
    void process_visual_input();
    
    /**
     * @brief Update working memory with current context
     */
    void update_working_memory();
    
    /**
     * @brief Select and execute action based on current state
     */
    void select_and_execute_action();
    
    /**
     * @brief Learn from recent experience
     */
    void learn_from_experience();
    
    // ========================================================================
    // ACTION GENERATION AND EXECUTION
    // ========================================================================
    
    /**
     * @brief Generate possible actions based on current context
     * @return Vector of possible actions  
     */
    std::vector<BrowsingAction> generate_action_candidates();
    
    /**
     * @brief Execute a specific action
     * @param action Action to execute
     */
    void execute_action(const BrowsingAction& action);
    
    /**
     * @brief Calculate reward for immediate action
     * @return Reward value
     */
    float calculate_immediate_reward();
    
    // ========================================================================
    // ENVIRONMENT INTERACTION
    // ========================================================================
    
    /**
     * @brief Set environment sensor function
     * @param sensor Function that returns environmental state
     */
    void setEnvironmentSensor(std::function<BrowsingState()> sensor);
    void setActionExecutor(std::function<void(const BrowsingAction&)> executor);

    // Public methods for agent control
    void start();
    void stop();
    void run();
    void set_learning_goal(const std::string& goal);
    bool isActionValid(const BrowsingAction& action);

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Configuration
    NetworkConfig config_;
    OperatingMode current_mode_;
    bool is_learning_active_;
    bool detailed_logging_;
    bool is_passive_mode_;
    float simulation_time_;
    std::chrono::steady_clock::time_point last_action_time_;
    std::mt19937 gen;
    std::string save_path_;
    std::function<BrowsingState()> environment_sensor_;
    std::function<void(const BrowsingAction&)> action_executor_;

    // New member variables
    std::vector<std::string> learning_goals_;
    std::vector<float> environmental_context_;
    std::vector<float> current_goals_;
    BrowsingAction selected_action_;
    float exploration_rate_;
    std::vector<float> global_state_;
    float global_reward_signal_;

    // Core Components
    std::unique_ptr<ControllerModule> controller_module_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::unique_ptr<InputController> input_controller_;
    std::unique_ptr<BrainModuleArchitecture> brain_architecture_;
    std::unordered_map<std::string, std::unique_ptr<SpecializedModule>> modules_;
    std::unordered_map<std::string, int> module_neuron_counts_;  // Track neuron counts per module

    struct AgentMetrics {
        int total_actions = 0;
        int successful_actions = 0;
        float average_reward = 0.0f;
    };

    AgentMetrics metrics_;

    // Token generation state
    static constexpr int VOCAB_SIZE = 32000;
    mutable std::vector<std::vector<float>> output_embedding_weights_;  // Neural output -> token logits projection
    mutable bool output_layer_initialized_ = false;
    mutable std::vector<int> last_generated_tokens_;  // Store last generated token IDs

    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    void initialize_neural_modules();
    void initialize_attention_system();
    void update_learning_goals();
    void log_action(const std::string& action);
    void initializeSpecializedModules();
    void setupDefaultLearningGoals();
    
    // Real screen-based reinforcement learning methods
    void processRealScreenInput();
    void executeRealAction();
    float computeScreenBasedReward();
    float evaluateGoalProgress();
    float evaluateExplorationEffectiveness();
    float evaluateActionPenalties();
    float evaluateLearningEfficiency();
    float evaluateTaskCompletion();
    float evaluateLearningImprovement();
    void learnFromActionOutcome(float reward);
    void storeEpisodeInMemory(float reward);
    void logLearningProgress(int step, float reward);
    
    // Missing method declarations needed by DecisionAndActionSystems.cpp
    void update_attention_weights();
    void coordinate_modules();
    std::vector<float> collect_inter_module_signals(const std::string& target_module);
    void distribute_module_output(const std::string& source_module, const std::vector<float>& output);
    void make_decision();
    std::vector<BrowsingAction> translate_neural_output_to_actions(const std::vector<float>& neural_output);
    std::vector<BrowsingAction> generate_base_action_candidates();
    std::vector<float> evaluate_action_candidates(const std::vector<BrowsingAction>& candidates, 
                                                   const std::vector<MemorySystem::MemoryTrace>& similar_episodes);
    std::vector<BrowsingAction> generate_action_candidates_for_goal(const std::vector<float>& goal);
    void select_action_with_exploration(const std::vector<BrowsingAction>& candidates, 
                                                   const std::vector<float>& values);
    void execute_action();
    void execute_click_action();
    void execute_scroll_action();
    void execute_type_action();
    void execute_enter_action();
    void execute_backspace_action();
    std::vector<float> convert_action_to_motor_command(const BrowsingAction& action);
    void learn_from_feedback();
    float compute_action_reward();
    void adapt_exploration_rate();
    void apply_modular_learning(float reward);
    void update_global_state();
    void consolidate_learning();
    void transfer_knowledge_between_modules();

    // Helper methods for persistence and neural network management
    bool saveModule(const std::string& module_name, const std::string& save_path);
    bool loadModule(const std::string& module_name, const std::string& load_path);
    int getTotalNeuronCount() const;
    int getModuleNeuronCount(const std::string& module_name) const;
    std::string getCurrentTimestamp() const;
    
    // Language processing helper methods
    std::vector<float> extractLanguageFeatures(const std::string& text) const;
    float computeLanguageComprehension(const std::vector<float>& neural_output) const;
    std::string convertNeuralToLanguage(const std::vector<float>& neural_features) const;

    // Token generation methods
    std::vector<float> computeTokenLogits(const std::vector<float>& neural_output) const;
    int sampleToken(const std::vector<float>& logits, float temperature = 1.0f) const;
    std::vector<int> generateTokenSequence(const std::vector<float>& neural_output, int max_tokens = 20) const;
    std::string decodeTokenSequence(const std::vector<int>& token_ids) const;

    /**
     * @brief Update exploration rate based on performance
     */
    float updateExplorationRate(float current_rate, float recent_performance, 
                               float target_performance = 0.8f);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert ActionType to string
 */
inline std::string actionTypeToString(ActionType type) {
    switch (type) {
        case ActionType::CLICK: return "CLICK";
        case ActionType::SCROLL: return "SCROLL";
        case ActionType::TYPE: return "TYPE";
        case ActionType::ENTER: return "ENTER";
        case ActionType::BACKSPACE: return "BACKSPACE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert string to ActionType
 */
inline ActionType stringToActionType(const std::string& type_str) {
    if (type_str == "CLICK") return ActionType::CLICK;
    if (type_str == "SCROLL") return ActionType::SCROLL;
    if (type_str == "TYPE") return ActionType::TYPE;
    if (type_str == "ENTER") return ActionType::ENTER;
    if (type_str == "BACKSPACE") return ActionType::BACKSPACE;
    return ActionType::CLICK; // default
}

/**
 * @brief Compute similarity between two browsing states
 */
float computeBrowsingStateSimilarity(const BrowsingState& state1, const BrowsingState& state2);

/**
 * @brief Compute action value using simple heuristics
 */
float computeActionValue(const BrowsingAction& action, const BrowsingState& state);

#endif // AUTONOMOUS_LEARNING_AGENT_H
