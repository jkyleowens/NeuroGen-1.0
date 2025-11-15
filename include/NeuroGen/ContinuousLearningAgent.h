// Continuous Learning Agent for NeuroGen
// Orchestrates autonomous learning across sessions with persistent state

#ifndef CONTINUOUS_LEARNING_AGENT_H
#define CONTINUOUS_LEARNING_AGENT_H

#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "NeuroGen/BrainModuleArchitecture.h"

/**
 * @brief Input data structure for multi-modal learning
 */
struct MultiModalInput {
    // Visual input
    std::vector<float> visual_data;
    int visual_width = 0;
    int visual_height = 0;
    int visual_channels = 0;
    
    // Text input
    std::string text_data;
    std::vector<float> text_embeddings;
    
    // Audio input (future extension)
    std::vector<float> audio_data;
    int audio_sample_rate = 0;
    
    // Metadata
    std::chrono::system_clock::time_point timestamp;
    std::string source_url;
    std::string content_type;
    float relevance_score = 1.0f;
    
    // Context information
    std::map<std::string, float> context_features;
    std::vector<std::string> tags;
};

/**
 * @brief Learning experience for reinforcement learning
 */
struct LearningExperience {
    std::vector<float> state;
    std::vector<float> action;
    float reward;
    std::vector<float> next_state;
    bool terminal;
    std::chrono::system_clock::time_point timestamp;
    std::string experience_type; // "web_browse", "read_text", "view_image", etc.
};

/**
 * @brief Autonomous agent that continuously learns across sessions
 */
class ContinuousLearningAgent {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor with configuration
     * @param config_path Path to agent configuration file
     */
    ContinuousLearningAgent(const std::string& config_path);
    
    /**
     * @brief Destructor with graceful shutdown
     */
    ~ContinuousLearningAgent();
    
    /**
     * @brief Initialize the agent and all subsystems
     * @return Success status
     */
    bool initialize();
    
    /**
     * @brief Start continuous learning mode
     * @param session_name Optional session name
     * @return Success status
     */
    bool startContinuousLearning(const std::string& session_name = "");
    
    /**
     * @brief Stop continuous learning and save state
     * @return Success status
     */
    bool stopContinuousLearning();
    
    // ========================================================================
    // LEARNING CONTROL
    // ========================================================================
    
    /**
     * @brief Process single learning step
     * @param input Multi-modal input data
     * @return Learning success status
     */
    bool processLearningStep(const MultiModalInput& input);
    
    /**
     * @brief Set learning mode
     * @param exploration_rate Rate of exploration vs exploitation (0-1)
     * @param curiosity_drive Strength of curiosity-driven learning (0-1)
     */
    void setLearningMode(float exploration_rate, float curiosity_drive);
    
    /**
     * @brief Add learning experience to replay buffer
     * @param experience Learning experience to store
     */
    void addExperience(const LearningExperience& experience);
    
    /**
     * @brief Trigger memory consolidation
     * @param strength Consolidation strength (0-1)
     */
    void triggerMemoryConsolidation(float strength = 0.1f);
    
    // ========================================================================
    // AUTONOMOUS BROWSING AND LEARNING
    // ========================================================================
    
    /**
     * @brief Start autonomous web browsing session
     * @param start_urls Initial URLs to explore
     * @param max_duration Maximum duration in minutes
     * @return Success status
     */
    bool startAutonomousBrowsing(const std::vector<std::string>& start_urls, 
                                int max_duration = 60);
    
    /**
     * @brief Process web page content for learning
     * @param url Page URL
     * @param content Page content (HTML, text, etc.)
     * @param media_elements Images, videos, etc.
     * @return Learning reward signal
     */
    float processWebContent(const std::string& url, 
                           const std::string& content,
                           const std::vector<MultiModalInput>& media_elements);
    
    /**
     * @brief Calculate curiosity-driven reward
     * @param input Input data
     * @return Curiosity reward
     */
    float calculateCuriosityReward(const MultiModalInput& input);
    
    // ========================================================================
    // MEMORY AND KNOWLEDGE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add information to long-term memory
     * @param content Content to memorize
     * @param importance Importance score (0-1)
     * @param tags Associated tags
     */
    void addToMemory(const std::string& content, float importance, 
                    const std::vector<std::string>& tags);
    
    /**
     * @brief Retrieve relevant memories
     * @param query Query for memory retrieval
     * @param max_results Maximum number of results
     * @return Relevant memory contents
     */
    std::vector<std::string> retrieveMemories(const std::string& query, 
                                             int max_results = 10);
    
    /**
     * @brief Update knowledge graph
     * @param entities Detected entities
     * @param relationships Entity relationships
     */
    void updateKnowledgeGraph(const std::vector<std::string>& entities,
                             const std::vector<std::pair<std::string, std::string>>& relationships);
    
    // ========================================================================
    // ATTENTION AND GOAL MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Set current learning goal
     * @param goal_description Description of the learning goal
     * @param priority Goal priority (0-1)
     */
    void setLearningGoal(const std::string& goal_description, float priority = 1.0f);
    
    /**
     * @brief Update attention based on current context
     * @param context_vector Current context representation
     */
    void updateAttention(const std::vector<float>& context_vector);
    
    /**
     * @brief Get current attention distribution across modules
     * @return Map of module names to attention weights
     */
    std::map<std::string, float> getAttentionDistribution() const;
    
    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get learning statistics
     * @return Map of statistics
     */
    std::map<std::string, float> getLearningStatistics() const;
    
    /**
     * @brief Get performance metrics over time
     * @param time_window Time window in minutes
     * @return Performance data
     */
    std::vector<std::pair<std::chrono::system_clock::time_point, float>> 
        getPerformanceOverTime(int time_window = 60) const;
    
    /**
     * @brief Check if agent is learning effectively
     * @return True if learning progress is detected
     */
    bool isLearningEffectively() const;
    
    // ========================================================================
    // SESSION MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save current session
     * @param checkpoint_name Optional checkpoint name
     * @return Success status
     */
    bool saveSession(const std::string& checkpoint_name = "");
    
    /**
     * @brief Load previous session
     * @param session_id Session to load
     * @return Success status
     */
    bool loadSession(const std::string& session_id);
    
    /**
     * @brief Get available sessions
     * @return List of session IDs with metadata
     */
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> 
        getAvailableSessions() const;

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Core components
    std::unique_ptr<BrainModuleArchitecture> brain_architecture_;
    std::unique_ptr<LearningStateManager> state_manager_;
    
    // Learning configuration
    std::string config_path_;
    float exploration_rate_ = 0.3f;
    float curiosity_drive_ = 0.5f;
    
    // Experience replay
    std::queue<LearningExperience> experience_buffer_;
    size_t max_buffer_size_ = 100000;
    
    // Continuous learning state
    std::atomic<bool> is_learning_ = false;
    std::atomic<bool> should_stop_ = false;
    std::thread learning_thread_;
    
    // Memory management
    std::vector<std::string> long_term_memory_;
    std::map<std::string, std::vector<std::string>> memory_index_;
    
    // Knowledge graph (simplified)
    std::map<std::string, std::vector<std::string>> knowledge_graph_;
    
    // Performance tracking
    std::vector<std::pair<std::chrono::system_clock::time_point, float>> performance_history_;
    mutable std::mutex performance_mutex_;
    
    // Thread synchronization
    mutable std::mutex learning_mutex_;
    mutable std::mutex memory_mutex_;
    std::condition_variable learning_cv_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    /**
     * @brief Main learning loop (runs in separate thread)
     */
    void learningLoop();
    
    /**
     * @brief Process experience replay batch
     * @param batch_size Size of the experience batch
     */
    void processExperienceReplay(size_t batch_size = 32);
    
    /**
     * @brief Update module attention weights
     * @param context Current context
     */
    void updateModuleAttention(const std::vector<float>& context);
    
    /**
     * @brief Calculate learning reward
     * @param input Input data
     * @param prediction_error Prediction error
     * @return Combined reward signal
     */
    float calculateLearningReward(const MultiModalInput& input, float prediction_error);
    
    /**
     * @brief Extract features from input
     * @param input Multi-modal input
     * @return Feature vector
     */
    std::vector<float> extractFeatures(const MultiModalInput& input);
    
    /**
     * @brief Update performance metrics
     * @param reward Current reward
     * @param prediction_error Current prediction error
     */
    void updatePerformanceMetrics(float reward, float prediction_error);
    
    /**
     * @brief Load configuration from file
     * @return Success status
     */
    bool loadConfiguration();
    
    /**
     * @brief Initialize brain architecture
     * @return Success status
     */
    bool initializeBrainArchitecture();
    
    /**
     * @brief Cleanup resources
     */
    void cleanup();
};

// ============================================================================
// IMPLEMENTATION OF KEY METHODS
// ============================================================================

inline bool ContinuousLearningAgent::processLearningStep(const MultiModalInput& input) {
    if (!is_learning_) {
        return false;
    }
    
    try {
        // Extract features from multi-modal input
        std::vector<float> features = extractFeatures(input);
        
        // Process through brain architecture
        auto module_outputs = brain_architecture_->processInput(features);
        
        // Calculate prediction error (simplified)
        float prediction_error = 0.0f;
        for (const auto& [module_name, output] : module_outputs) {
            // Compare output with expected (this would be more sophisticated in practice)
            prediction_error += std::accumulate(output.begin(), output.end(), 0.0f) / output.size();
        }
        prediction_error = std::abs(prediction_error - 0.5f); // Normalize around 0.5
        
        // Calculate reward
        float learning_reward = calculateLearningReward(input, prediction_error);
        float curiosity_reward = calculateCuriosityReward(input);
        float total_reward = learning_reward + curiosity_drive_ * curiosity_reward;
        
        // Apply learning
        brain_architecture_->applyLearning(total_reward, prediction_error);
        
        // Update state manager
        for (const auto& [module_name, _] : module_outputs) {
            state_manager_->updatePerformanceMetrics(module_name, total_reward, prediction_error);
        }
        
        // Create learning experience
        LearningExperience experience;
        experience.state = features;
        experience.reward = total_reward;
        experience.timestamp = std::chrono::system_clock::now();
        experience.experience_type = input.content_type;
        
        addExperience(experience);
        
        // Update performance tracking
        updatePerformanceMetrics(total_reward, prediction_error);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in learning step: " << e.what() << std::endl;
        return false;
    }
}

inline void ContinuousLearningAgent::learningLoop() {
    while (!should_stop_) {
        if (is_learning_) {
            // Process experience replay
            if (experience_buffer_.size() >= 32) {
                processExperienceReplay(32);
            }
            
            // Periodic memory consolidation
            static auto last_consolidation = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            auto minutes_elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - last_consolidation).count();
            
            if (minutes_elapsed >= 30) { // Consolidate every 30 minutes
                triggerMemoryConsolidation(0.05f);
                last_consolidation = now;
            }
            
            // Periodic state saving
            static auto last_save = std::chrono::steady_clock::now();
            auto save_minutes = std::chrono::duration_cast<std::chrono::minutes>(now - last_save).count();
            
            if (save_minutes >= 10) { // Save every 10 minutes
                saveSession("auto_checkpoint");
                last_save = now;
            }
        }
        
        // Sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

#endif // CONTINUOUS_LEARNING_AGENT_H