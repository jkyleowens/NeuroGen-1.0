// ============================================================================
// DECISION AND ACTION SYSTEMS HEADER
// File: include/NeuroGen/DecisionAndActionSystems.h
// ============================================================================

#ifndef DECISION_AND_ACTION_SYSTEMS_H
#define DECISION_AND_ACTION_SYSTEMS_H

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <chrono>

// Forward declarations
class SpecializedModule;
class AutonomousLearningAgent;
class MemorySystem;
class AttentionController;

namespace DecisionAndActionSystems {

// ============================================================================
// DECISION STRUCTURES AND ENUMS
// ============================================================================

/**
 * @brief Types of decisions the agent can make
 */
enum class DecisionType {
    GENERATE_RESPONSE,        // Generate a language response
    SEEK_MORE_INFORMATION,    // Request additional input or context
    CONSOLIDATE_MEMORY,       // Trigger memory consolidation
    WAIT_AND_OBSERVE,         // Continue processing current context
    ADJUST_ATTENTION,         // Modify attention allocation
    LEARN_FROM_FEEDBACK,      // Process feedback signals
    EXPLORE_ALTERNATIVES,     // Try different approaches
    MAINTAIN_CURRENT_STATE    // Keep current processing state
};

/**
 * @brief Decision context structure
 */
struct DecisionContext {
    std::vector<float> language_comprehension_state;
    std::vector<float> semantic_memory_state;
    std::vector<float> working_memory_state;
    std::vector<float> current_goals;
    std::vector<float> environmental_context;
    float global_reward_signal;
    float simulation_time;
    float decision_urgency;
    std::string current_task_type;
    std::map<std::string, float> module_activities;
    
    DecisionContext() : global_reward_signal(0.0f), simulation_time(0.0f), decision_urgency(0.5f) {}
};

/**
 * @brief Decision outcome structure
 */
struct DecisionOutcome {
    DecisionType decision_type;
    std::string decision_description;
    float confidence_level;
    std::vector<float> action_parameters;
    std::map<std::string, float> expected_outcomes;
    float estimated_reward;
    std::chrono::steady_clock::time_point timestamp;
    
    DecisionOutcome() : decision_type(DecisionType::WAIT_AND_OBSERVE), 
                       confidence_level(0.0f), estimated_reward(0.0f) {
        timestamp = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Action execution result
 */
struct ActionResult {
    bool success;
    std::string result_description;
    float actual_reward;
    std::vector<float> state_changes;
    std::map<std::string, float> performance_metrics;
    std::chrono::steady_clock::time_point execution_time;
    
    ActionResult() : success(false), actual_reward(0.0f) {
        execution_time = std::chrono::steady_clock::now();
    }
};

// ============================================================================
// DECISION MAKING INTERFACES
// ============================================================================

/**
 * @brief Abstract decision maker interface
 */
class DecisionMaker {
public:
    virtual ~DecisionMaker() = default;
    
    /**
     * @brief Make a decision based on current context
     * @param context Current decision context
     * @return Decision outcome
     */
    virtual DecisionOutcome makeDecision(const DecisionContext& context) = 0;
    
    /**
     * @brief Update decision maker with feedback
     * @param outcome Previous decision outcome
     * @param result Actual result of action
     */
    virtual void updateWithFeedback(const DecisionOutcome& outcome, const ActionResult& result) = 0;
    
    /**
     * @brief Get decision maker type
     * @return Type description string
     */
    virtual std::string getType() const = 0;
};

/**
 * @brief Abstract action executor interface
 */
class ActionExecutor {
public:
    virtual ~ActionExecutor() = default;
    
    /**
     * @brief Execute an action based on decision
     * @param decision Decision to execute
     * @param agent Reference to autonomous learning agent
     * @return Action execution result
     */
    virtual ActionResult executeAction(const DecisionOutcome& decision, 
                                     AutonomousLearningAgent& agent) = 0;
    
    /**
     * @brief Check if action type is supported
     * @param decision_type Type of decision/action
     * @return True if supported
     */
    virtual bool supportsActionType(DecisionType decision_type) const = 0;
    
    /**
     * @brief Get executor type
     * @return Type description string
     */
    virtual std::string getType() const = 0;
};

// ============================================================================
// SPECIALIZED PROCESSING FUNCTIONS
// ============================================================================

/**
 * @brief Namespace for specialized module processing functions
 */
namespace SpecializedProcessing {
    
    /**
     * @brief Process executive function module
     * @param module Reference to specialized module
     * @param executive_input Input for executive processing
     * @return Processed output vector
     */
    std::vector<float> processExecutiveFunction(SpecializedModule& module,
                                              const std::vector<float>& executive_input);
    
    /**
     * @brief Process working memory module
     * @param module Reference to specialized module
     * @param memory_input Input for memory processing
     * @return Processed output vector
     */
    std::vector<float> processWorkingMemory(SpecializedModule& module,
                                          const std::vector<float>& memory_input);
    
    /**
     * @brief Process reward system module
     * @param module Reference to specialized module
     * @param reward_input Input for reward processing
     * @return Processed output vector
     */
    std::vector<float> processRewardSystem(SpecializedModule& module,
                                         const std::vector<float>& reward_input);
    
    /**
     * @brief Process attention system module
     * @param module Reference to specialized module
     * @param attention_input Input for attention processing
     * @return Processed output vector
     */
    std::vector<float> processAttentionSystem(SpecializedModule& module,
                                            const std::vector<float>& attention_input);
    
    /**
     * @brief Process motor cortex module (adapted for text output)
     * @param module Reference to specialized module
     * @param motor_input Input for motor processing
     * @return Processed output vector
     */
    std::vector<float> processMotorCortex(SpecializedModule& module,
                                        const std::vector<float>& motor_input);
}

// ============================================================================
// LANGUAGE PROCESSING FUNCTIONS
// ============================================================================

/**
 * @brief Namespace for language-specific processing functions
 */
namespace LanguageProcessing {
    
    /**
     * @brief Extract linguistic features from text
     * @param text Input text to analyze
     * @return Vector of linguistic features
     */
    std::vector<float> extractLinguisticFeatures(const std::string& text);
    
    /**
     * @brief Convert decision to natural language description
     * @param decision Decision outcome to describe
     * @return Human-readable description
     */
    std::string decisionToLanguage(const DecisionOutcome& decision);
    
    /**
     * @brief Analyze text complexity for decision making
     * @param text Input text
     * @return Complexity metrics
     */
    std::map<std::string, float> analyzeTextComplexity(const std::string& text);
    
    /**
     * @brief Generate language response based on context
     * @param context Decision context
     * @param agent Reference to learning agent
     * @return Generated response text
     */
    std::string generateLanguageResponse(const DecisionContext& context,
                                       AutonomousLearningAgent& agent);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Create decision context from agent state
 * @param agent Reference to autonomous learning agent
 * @return Populated decision context
 */
DecisionContext createDecisionContext(const AutonomousLearningAgent& agent);

/**
 * @brief Calculate decision confidence based on module states
 * @param context Decision context
 * @return Confidence value [0.0-1.0]
 */
float calculateDecisionConfidence(const DecisionContext& context);

/**
 * @brief Update global reward signal based on action results
 * @param agent Reference to autonomous learning agent
 * @param result Action execution result
 */
void updateGlobalReward(AutonomousLearningAgent& agent, const ActionResult& result);

/**
 * @brief Log decision and action for analysis
 * @param decision Decision that was made
 * @param result Result of action execution
 */
void logDecisionAction(const DecisionOutcome& decision, const ActionResult& result);

/**
 * @brief Convert decision type to string
 * @param decision_type Decision type enum
 * @return String representation
 */
std::string decisionTypeToString(DecisionType decision_type);

/**
 * @brief Convert string to decision type
 * @param type_string String representation
 * @return Decision type enum
 */
DecisionType stringToDecisionType(const std::string& type_string);

// ============================================================================
// PERFORMANCE METRICS
// ============================================================================

/**
 * @brief Decision and action performance metrics
 */
struct PerformanceMetrics {
    size_t total_decisions_made;
    size_t successful_actions;
    float average_decision_confidence;
    float average_action_reward;
    std::map<DecisionType, size_t> decision_type_counts;
    std::map<std::string, float> module_utilization_rates;
    std::chrono::steady_clock::time_point last_update;
    
    PerformanceMetrics() : total_decisions_made(0), successful_actions(0),
                          average_decision_confidence(0.0f), average_action_reward(0.0f) {
        last_update = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Get current performance metrics
 * @param agent Reference to autonomous learning agent
 * @return Current performance metrics
 */
PerformanceMetrics getPerformanceMetrics(const AutonomousLearningAgent& agent);

/**
 * @brief Reset performance metrics
 * @param agent Reference to autonomous learning agent
 */
void resetPerformanceMetrics(AutonomousLearningAgent& agent);

} // namespace DecisionAndActionSystems

#endif // DECISION_AND_ACTION_SYSTEMS_H