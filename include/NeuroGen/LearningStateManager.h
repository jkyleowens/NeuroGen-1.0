// Update 4: Learning State Manager Implementation
// File: include/NeuroGen/LearningStateManager.h (NEW FILE)

#ifndef LEARNING_STATE_MANAGER_H
#define LEARNING_STATE_MANAGER_H

#include "NeuroGen/LearningState.h"
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <vector>
#include <string>
#include <chrono>
#include <map>

// Forward declaration to break circular dependency
class BrainModuleArchitecture;

/**
 * @brief Manages persistent learning state across sessions
 */
class LearningStateManager {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor with architecture reference
     */
    LearningStateManager(std::shared_ptr<BrainModuleArchitecture> architecture, 
                        const std::string& base_save_path);
    
    /**
     * @brief Destructor with automatic cleanup
     */
    ~LearningStateManager();
    
    /**
     * @brief Initialize the learning state manager
     */
    bool initialize();
    
    // ========================================================================
    // SESSION MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Start a new learning session
     */
    std::string startSession(const std::string& session_name = "");
    
    /**
     * @brief End current session and save state
     */
    bool endSession();
    
    /**
     * @brief Resume from a previous session
     */
    bool resumeSession(const std::string& session_id);
    
    /**
     * @brief Get list of available sessions
     */
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> getAvailableSessions() const;
    
    // ========================================================================
    // LEARNING STATE PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save current learning state
     */
    bool saveLearningState(const std::string& checkpoint_name = "");
    
    /**
     * @brief Load learning state
     */
    bool loadLearningState(const std::string& checkpoint_name = "");
    
    /**
     * @brief Auto-save learning state periodically
     */
    void enableAutoSave(bool enable, int interval_minutes = 10);
    
    // ========================================================================
    // MEMORY CONSOLIDATION
    // ========================================================================
    
    /**
     * @brief Perform memory consolidation
     */
    bool performMemoryConsolidation(float consolidation_strength = 0.1f);
    
    /**
     * @brief Schedule automatic consolidation
     */
    void enableAutoConsolidation(bool enable, int interval_minutes = 30);
    
    // ========================================================================
    // PERFORMANCE TRACKING
    // ========================================================================
    
    /**
     * @brief Update learning performance metrics
     */
    void updatePerformanceMetrics(const std::string& module_name, 
                                 float reward, float prediction_error);
    
    /**
     * @brief Get current learning statistics
     */
    std::map<std::string, float> getLearningStatistics() const;
    
    /**
     * @brief Check if learning is progressing
     */
    bool isLearningProgressing() const;

private:
    // Core components
    std::shared_ptr<BrainModuleArchitecture> architecture_;
    std::string base_save_path_;
    std::string current_session_id_;
    
    // Session state
    SessionLearningState current_session_state_;
    
    // Auto-save and consolidation
    std::atomic<bool> auto_save_enabled_{false};
    std::atomic<bool> auto_consolidation_enabled_{false};
    std::atomic<int> auto_save_interval_{10};
    std::atomic<int> auto_consolidation_interval_{30};
    
    // Background thread management
    std::atomic<bool> background_running_{false};
    std::thread background_thread_;
    mutable std::mutex state_mutex_;
    std::condition_variable background_cv_;
    
    // Performance tracking
    std::map<std::string, std::vector<float>> performance_history_;
    std::map<std::string, float> module_learning_rates_;
    
    // Internal methods
    void backgroundWorker();
    std::string generateSessionId() const;
    std::string getSessionDirectory(const std::string& session_id) const;
    bool createSessionDirectory(const std::string& session_id) const;
};

#endif // LEARNING_STATE_MANAGER_H