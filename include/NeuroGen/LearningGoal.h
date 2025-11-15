// ============================================================================
// LEARNING GOAL - AUTONOMOUS LEARNING OBJECTIVES
// File: include/NeuroGen/LearningGoal.h
// ============================================================================

#ifndef LEARNING_GOAL_H
#define LEARNING_GOAL_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <chrono>
#include <memory>

/**
 * @brief Base class for autonomous learning goals
 * 
 * Defines objectives and success criteria for the autonomous learning agent.
 * Goals can be language-specific (comprehension, generation) or general
 * cognitive objectives (memory consolidation, attention improvement).
 */
class AutonomousGoal {
public:
    enum class GoalType {
        LANGUAGE_COMPREHENSION,     // Improve text understanding
        LANGUAGE_GENERATION,        // Enhance response quality
        VOCABULARY_EXPANSION,       // Learn new words and concepts
        SEMANTIC_REASONING,         // Improve logical reasoning
        MEMORY_CONSOLIDATION,       // Strengthen memory patterns
        ATTENTION_OPTIMIZATION,     // Improve focus and resource allocation
        CONVERSATION_SKILLS,        // Enhance dialogue capabilities
        DOMAIN_SPECIALIZATION,      // Learn specific domain knowledge
        CREATIVE_EXPRESSION,        // Develop creative language abilities
        GENERAL_INTELLIGENCE        // Overall cognitive improvement
    };
    
    enum class GoalStatus {
        ACTIVE,           // Currently pursuing this goal
        COMPLETED,        // Goal has been achieved
        PAUSED,          // Temporarily suspended
        FAILED,          // Goal could not be achieved
        ABANDONED        // No longer pursuing this goal
    };

    // ========================================================================
    // CONSTRUCTION AND CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Constructor for autonomous goal
     * @param name Human-readable goal name
     * @param type Type of learning goal
     * @param priority Goal priority [0.0, 1.0]
     */
    AutonomousGoal(const std::string& name, GoalType type, float priority = 0.5f);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~AutonomousGoal() = default;
    
    /**
     * @brief Set goal parameters
     * @param parameters Map of parameter names to values
     */
    void setParameters(const std::map<std::string, float>& parameters);
    
    /**
     * @brief Set success criteria for the goal
     * @param criteria Map of criterion names to target values
     */
    void setSuccessCriteria(const std::map<std::string, float>& criteria);

    // ========================================================================
    // GOAL MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Update goal progress based on current performance
     * @param performance_metrics Current agent performance metrics
     * @return Updated progress value [0.0, 1.0]
     */
    virtual float updateProgress(const std::map<std::string, float>& performance_metrics);
    
    /**
     * @brief Check if goal has been completed
     * @param current_metrics Current performance metrics
     * @return True if goal completion criteria are met
     */
    virtual bool isCompleted(const std::map<std::string, float>& current_metrics) const;
    
    /**
     * @brief Get specific learning actions for this goal
     * @param current_state Current agent state
     * @return Vector of recommended learning actions
     */
    virtual std::vector<std::string> getRecommendedActions(
        const std::map<std::string, float>& current_state) const;
    
    /**
     * @brief Calculate reward signal for progress toward this goal
     * @param previous_metrics Previous performance metrics
     * @param current_metrics Current performance metrics
     * @return Reward signal [-1.0, 1.0]
     */
    virtual float calculateReward(const std::map<std::string, float>& previous_metrics,
                                const std::map<std::string, float>& current_metrics) const;

    // ========================================================================
    // GETTERS AND SETTERS
    // ========================================================================
    
    std::string getName() const { return name_; }
    GoalType getType() const { return type_; }
    GoalStatus getStatus() const { return status_; }
    float getPriority() const { return priority_; }
    float getProgress() const { return progress_; }
    
    void setStatus(GoalStatus status) { status_ = status; }
    void setPriority(float priority) { priority_ = std::max(0.0f, std::min(1.0f, priority)); }
    
    /**
     * @brief Get time elapsed since goal creation
     * @return Duration since goal was created
     */
    std::chrono::duration<float> getElapsedTime() const;
    
    /**
     * @brief Get goal description
     * @return Human-readable description of the goal
     */
    virtual std::string getDescription() const;
    
    /**
     * @brief Get current goal parameters
     * @return Map of parameter names to values
     */
    std::map<std::string, float> getParameters() const { return parameters_; }
    
    /**
     * @brief Get success criteria
     * @return Map of criterion names to target values
     */
    std::map<std::string, float> getSuccessCriteria() const { return success_criteria_; }

public:
    // Goal state (made public for simple access)
    bool is_active;
    float target_score;
    std::string description;

protected:
    // Core goal properties
    std::string name_;
    GoalType type_;
    GoalStatus status_;
    float priority_;
    float progress_;
    
    // Goal configuration
    std::map<std::string, float> parameters_;
    std::map<std::string, float> success_criteria_;
    
    // Timing information
    std::chrono::high_resolution_clock::time_point creation_time_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    // Progress tracking
    std::vector<float> progress_history_;
    float best_progress_;
    size_t update_count_;

    // ========================================================================
    // HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Calculate progress based on metric improvement
     * @param metric_name Name of the metric to track
     * @param current_value Current metric value
     * @param target_value Target metric value
     * @return Progress toward target [0.0, 1.0]
     */
    float calculateMetricProgress(const std::string& metric_name, 
                                float current_value, 
                                float target_value) const;
    
    /**
     * @brief Update internal progress tracking
     * @param new_progress New progress value
     */
    void updateInternalProgress(float new_progress);
};

// ============================================================================
// SPECIALIZED GOAL TYPES
// ============================================================================

/**
 * @brief Language comprehension improvement goal
 */
class LanguageComprehensionGoal : public AutonomousGoal {
public:
    LanguageComprehensionGoal(float target_comprehension_score = 0.8f);
    
    float updateProgress(const std::map<std::string, float>& performance_metrics) override;
    bool isCompleted(const std::map<std::string, float>& current_metrics) const override;
    std::vector<std::string> getRecommendedActions(
        const std::map<std::string, float>& current_state) const override;
    std::string getDescription() const override;
};

/**
 * @brief Language generation quality improvement goal
 */
class LanguageGenerationGoal : public AutonomousGoal {
public:
    LanguageGenerationGoal(float target_generation_quality = 0.8f);
    
    float updateProgress(const std::map<std::string, float>& performance_metrics) override;
    bool isCompleted(const std::map<std::string, float>& current_metrics) const override;
    std::vector<std::string> getRecommendedActions(
        const std::map<std::string, float>& current_state) const override;
    std::string getDescription() const override;
};

/**
 * @brief Vocabulary expansion goal
 */
class VocabularyExpansionGoal : public AutonomousGoal {
public:
    VocabularyExpansionGoal(size_t target_vocabulary_size = 10000);
    
    float updateProgress(const std::map<std::string, float>& performance_metrics) override;
    bool isCompleted(const std::map<std::string, float>& current_metrics) const override;
    std::vector<std::string> getRecommendedActions(
        const std::map<std::string, float>& current_state) const override;
    std::string getDescription() const override;

private:
    size_t target_vocabulary_size_;
    size_t initial_vocabulary_size_;
};

/**
 * @brief Conversation skills improvement goal
 */
class ConversationSkillsGoal : public AutonomousGoal {
public:
    ConversationSkillsGoal(float target_conversation_quality = 0.8f);
    
    float updateProgress(const std::map<std::string, float>& performance_metrics) override;
    bool isCompleted(const std::map<std::string, float>& current_metrics) const override;
    std::vector<std::string> getRecommendedActions(
        const std::map<std::string, float>& current_state) const override;
    std::string getDescription() const override;
};

// ============================================================================
// GOAL FACTORY
// ============================================================================

/**
 * @brief Factory for creating autonomous learning goals
 */
class GoalFactory {
public:
    /**
     * @brief Create a goal by type
     * @param type Type of goal to create
     * @param parameters Goal-specific parameters
     * @return Unique pointer to created goal
     */
    static std::unique_ptr<AutonomousGoal> createGoal(
        AutonomousGoal::GoalType type,
        const std::map<std::string, float>& parameters = {});
    
    /**
     * @brief Create a goal by name
     * @param goal_name String name of the goal
     * @param parameters Goal-specific parameters
     * @return Unique pointer to created goal
     */
    static std::unique_ptr<AutonomousGoal> createGoal(
        const std::string& goal_name,
        const std::map<std::string, float>& parameters = {});
    
    /**
     * @brief Get list of available goal types
     * @return Vector of goal type names
     */
    static std::vector<std::string> getAvailableGoalTypes();
};

#endif // LEARNING_GOAL_H