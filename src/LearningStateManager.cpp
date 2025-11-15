// File: src/LearningStateManager.cpp

#include "NeuroGen/LearningStateManager.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

LearningStateManager::LearningStateManager(std::shared_ptr<BrainModuleArchitecture> architecture, 
                                          const std::string& base_save_path)
    : architecture_(architecture), base_save_path_(base_save_path) {
    
    // Ensure base directory exists
    std::filesystem::create_directories(base_save_path_);
    
    std::cout << "🧠 Learning State Manager initialized with path: " << base_save_path_ << std::endl;
}

LearningStateManager::~LearningStateManager() {
    // Stop background thread
    background_running_ = false;
    background_cv_.notify_all();
    
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
    
    // Auto-save current session if active
    if (!current_session_id_.empty()) {
        std::cout << "🔄 Auto-saving session on shutdown..." << std::endl;
        saveLearningState("shutdown_auto_save");
    }
}

bool LearningStateManager::initialize() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    try {
        // Initialize performance tracking for all modules
        auto module_names = architecture_->getModuleNames();
        for (const auto& name : module_names) {
            performance_history_[name].reserve(10000);
            module_learning_rates_[name] = 0.001f;
        }
        
        // Start background worker thread
        background_running_ = true;
        background_thread_ = std::thread(&LearningStateManager::backgroundWorker, this);
        
        std::cout << "✅ Learning State Manager initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize Learning State Manager: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

std::string LearningStateManager::startSession(const std::string& session_name) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Generate unique session ID
    current_session_id_ = generateSessionId();
    if (!session_name.empty()) {
        current_session_id_ += "_" + session_name;
    }
    
    // Create session directory
    if (!createSessionDirectory(current_session_id_)) {
        std::cerr << "❌ Failed to create session directory" << std::endl;
        current_session_id_.clear();
        return "";
    }
    
    // Initialize session state
    current_session_state_ = SessionLearningState();
    current_session_state_.session_id = current_session_id_;
    current_session_state_.session_start = std::chrono::system_clock::now();
    current_session_state_.architecture_hash = architecture_->calculateArchitectureHash();
    
    auto stats = architecture_->getArchitectureStatistics();
    current_session_state_.total_modules = stats["total_modules"];
    current_session_state_.total_neurons = stats["total_neurons"];
    current_session_state_.total_synapses = stats["total_synapses"];
    
    std::cout << "🚀 Started new learning session: " << current_session_id_ << std::endl;
    return current_session_id_;
}

bool LearningStateManager::endSession() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (current_session_id_.empty()) {
        std::cerr << "❌ No active session to end" << std::endl;
        return false;
    }
    
    try {
        // Save final session state
        bool success = saveLearningState("session_end");
        
        // Perform final consolidation
        performMemoryConsolidation(0.2f);
        
        // Save session summary
        std::string summary_path = getSessionDirectory(current_session_id_) + "/session_summary.txt";
        std::ofstream summary(summary_path);
        if (summary.is_open()) {
            auto session_duration = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::system_clock::now() - current_session_state_.session_start);
            
            summary << "Session Summary\n";
            summary << "===============\n";
            summary << "Session ID: " << current_session_id_ << "\n";
            summary << "Duration: " << session_duration.count() << " minutes\n";
            summary << "Total Learning Steps: " << current_session_state_.total_learning_steps << "\n";
            summary << "Cumulative Reward: " << current_session_state_.cumulative_reward << "\n";
            summary << "Average Performance: " << current_session_state_.average_performance << "\n";
            summary << "Consolidation Cycles: " << current_session_state_.consolidation_cycles << "\n";
            
            // Module-specific performance
            summary << "\nModule Performance:\n";
            for (const auto& [module_name, history] : performance_history_) {
                if (!history.empty()) {
                    float avg = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
                    summary << "  " << module_name << ": " << avg << "\n";
                }
            }
            
            summary.close();
        }
        
        std::cout << "✅ Ended session: " << current_session_id_ << std::endl;
        current_session_id_.clear();
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error ending session: " << e.what() << std::endl;
        return false;
    }
}

bool LearningStateManager::resumeSession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Verify session exists
    std::string session_dir = getSessionDirectory(session_id);
    if (!std::filesystem::exists(session_dir)) {
        std::cerr << "❌ Session directory does not exist: " << session_dir << std::endl;
        return false;
    }
    
    // Set current session
    current_session_id_ = session_id;
    
    // Load the latest checkpoint
    bool success = loadLearningState();
    
    if (success) {
        std::cout << "✅ Resumed session: " << session_id << std::endl;
        std::cout << "📊 Previous learning steps: " << current_session_state_.total_learning_steps << std::endl;
        std::cout << "🎯 Previous cumulative reward: " << current_session_state_.cumulative_reward << std::endl;
    } else {
        std::cerr << "❌ Failed to resume session: " << session_id << std::endl;
        current_session_id_.clear();
    }
    
    return success;
}

// ============================================================================
// LEARNING STATE PERSISTENCE
// ============================================================================

bool LearningStateManager::saveLearningState(const std::string& checkpoint_name) {
    if (current_session_id_.empty()) {
        std::cerr << "❌ No active session to save" << std::endl;
        return false;
    }
    
    try {
        // Update current session state from architecture
        current_session_state_ = architecture_->getGlobalLearningState();
        current_session_state_.session_id = current_session_id_;
        current_session_state_.last_checkpoint = std::chrono::system_clock::now();
        
        // Generate checkpoint name if not provided
        std::string checkpoint_id = checkpoint_name;
        if (checkpoint_id.empty()) {
            checkpoint_id = "checkpoint_" + std::to_string(current_session_state_.total_learning_steps);
        }
        
        // Save to session directory
        std::string session_dir = getSessionDirectory(current_session_id_);
        bool success = architecture_->saveLearningState(session_dir, checkpoint_id);
        
        if (success) {
            std::cout << "💾 Saved learning state: " << checkpoint_id << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving learning state: " << e.what() << std::endl;
        return false;
    }
}

bool LearningStateManager::loadLearningState(const std::string& checkpoint_name) {
    if (current_session_id_.empty()) {
        std::cerr << "❌ No active session to load into" << std::endl;
        return false;
    }
    
    try {
        std::string session_dir = getSessionDirectory(current_session_id_);
        bool success = architecture_->loadLearningState(session_dir, checkpoint_name);
        
        if (success) {
            // Update our session state
            current_session_state_ = architecture_->getGlobalLearningState();
            current_session_state_.session_id = current_session_id_;
            
            std::cout << "📁 Loaded learning state: " << checkpoint_name << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading learning state: " << e.what() << std::endl;
        return false;
    }
}

void LearningStateManager::enableAutoSave(bool enable, int interval_minutes) {
    auto_save_enabled_ = enable;
    auto_save_interval_ = interval_minutes;
    
    if (enable) {
        std::cout << "🔄 Auto-save enabled with " << interval_minutes << " minute intervals" << std::endl;
    } else {
        std::cout << "⏸️  Auto-save disabled" << std::endl;
    }
    
    // Wake up background thread to update timing
    background_cv_.notify_one();
}

// ============================================================================
// MEMORY CONSOLIDATION
// ============================================================================

bool LearningStateManager::performMemoryConsolidation(float consolidation_strength) {
    try {
        std::cout << "🧠 Starting memory consolidation (strength: " << consolidation_strength << ")" << std::endl;
        
        size_t total_consolidated = architecture_->performGlobalMemoryConsolidation(consolidation_strength);
        
        // Update session state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_session_state_.consolidation_cycles++;
            current_session_state_.last_consolidation = std::chrono::system_clock::now();
        }
        
        std::cout << "✅ Memory consolidation completed. Consolidated " << total_consolidated << " synapses" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during memory consolidation: " << e.what() << std::endl;
        return false;
    }
}

void LearningStateManager::enableAutoConsolidation(bool enable, int interval_minutes) {
    auto_consolidation_enabled_ = enable;
    auto_consolidation_interval_ = interval_minutes;
    
    if (enable) {
        std::cout << "🧠 Auto-consolidation enabled with " << interval_minutes << " minute intervals" << std::endl;
    } else {
        std::cout << "⏸️  Auto-consolidation disabled" << std::endl;
    }
    
    // Wake up background thread to update timing
    background_cv_.notify_one();
}

// ============================================================================
// PERFORMANCE TRACKING
// ============================================================================

void LearningStateManager::updatePerformanceMetrics(const std::string& module_name, 
                                                   float reward, float prediction_error) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Update performance history
    performance_history_[module_name].push_back(reward);
    
    // Limit history size
    if (performance_history_[module_name].size() > 10000) {
        performance_history_[module_name].erase(
            performance_history_[module_name].begin(),
            performance_history_[module_name].begin() + 1000
        );
    }
    
    // Update session state
    current_session_state_.total_learning_steps++;
    current_session_state_.cumulative_reward += reward;
    
    // Update average performance
    float total_reward = 0.0f;
    size_t total_samples = 0;
    for (const auto& [name, history] : performance_history_) {
        total_reward += std::accumulate(history.begin(), history.end(), 0.0f);
        total_samples += history.size();
    }
    if (total_samples > 0) {
        current_session_state_.average_performance = total_reward / total_samples;
    }
    
    // Adaptive learning rate
    float& learning_rate = module_learning_rates_[module_name];
    if (std::abs(prediction_error) > 0.5f) {
        learning_rate = std::min(learning_rate * 1.01f, 0.01f);
    } else {
        learning_rate = std::max(learning_rate * 0.999f, 0.0001f);
    }
}

std::map<std::string, float> LearningStateManager::getLearningStatistics() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::map<std::string, float> stats;
    
    stats["total_learning_steps"] = static_cast<float>(current_session_state_.total_learning_steps);
    stats["cumulative_reward"] = current_session_state_.cumulative_reward;
    stats["average_performance"] = current_session_state_.average_performance;
    stats["consolidation_cycles"] = static_cast<float>(current_session_state_.consolidation_cycles);
    
    // Calculate session duration in minutes
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - current_session_state_.session_start);
    stats["session_duration_minutes"] = static_cast<float>(duration.count());
    
    // Module-specific statistics
    for (const auto& [module_name, history] : performance_history_) {
        if (!history.empty()) {
            float avg = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
            stats["module_" + module_name + "_avg_performance"] = avg;
            stats["module_" + module_name + "_learning_rate"] = module_learning_rates_.at(module_name);
        }
    }
    
    return stats;
}

bool LearningStateManager::isLearningProgressing() const {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (current_session_state_.total_learning_steps < 1000) {
        return true; // Not enough data yet
    }

    float recent_performance = 0.0f;
    size_t recent_samples = 0;
    float historical_performance = 0.0f;
    size_t historical_samples = 0;

    for (const auto& [module_name, history] : performance_history_) {
        if (history.size() > 100) {
            // Recent 10%
            size_t recent_count = history.size() / 10;
            recent_performance += std::accumulate(history.end() - recent_count, history.end(), 0.0f);
            recent_samples += recent_count;

            // Historical 90%
            historical_performance += std::accumulate(history.begin(), history.end() - recent_count, 0.0f);
            historical_samples += history.size() - recent_count;
        }
    }

    if (recent_samples == 0 || historical_samples == 0) {
        return true; // Not enough data
    }

    return (recent_performance / recent_samples) > (historical_performance / historical_samples);
}

// ============================================================================
// BACKGROUND WORKER IMPLEMENTATION
// ============================================================================

void LearningStateManager::backgroundWorker() {
    auto last_save = std::chrono::steady_clock::now();
    auto last_consolidation = std::chrono::steady_clock::now();
    
    while (background_running_) {
        std::unique_lock<std::mutex> lock(state_mutex_);
        
        // Wait for either timeout or notification
        background_cv_.wait_for(lock, std::chrono::minutes(1));
        
        if (!background_running_) break;
        
        auto now = std::chrono::steady_clock::now();
        
        // Check for auto-save
        if (auto_save_enabled_ && !current_session_id_.empty()) {
            auto minutes_since_save = std::chrono::duration_cast<std::chrono::minutes>(now - last_save).count();
            if (minutes_since_save >= auto_save_interval_) {
                lock.unlock(); // Unlock before potentially long operation
                
                std::cout << "🔄 Auto-saving learning state..." << std::endl;
                if (saveLearningState("auto_save")) {
                    last_save = now;
                }
                
                lock.lock();
            }
        }
        
        // Check for auto-consolidation
        if (auto_consolidation_enabled_ && !current_session_id_.empty()) {
            auto minutes_since_consolidation = std::chrono::duration_cast<std::chrono::minutes>(now - last_consolidation).count();
            if (minutes_since_consolidation >= auto_consolidation_interval_) {
                lock.unlock(); // Unlock before potentially long operation
                
                if (performMemoryConsolidation(0.05f)) {
                    last_consolidation = now;
                }
                
                lock.lock();
            }
        }
    }
}

// ============================================================================
// HELPER METHODS
// ============================================================================

std::string LearningStateManager::generateSessionId() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "session_" << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string LearningStateManager::getSessionDirectory(const std::string& session_id) const {
    return base_save_path_ + "/" + session_id;
}

bool LearningStateManager::createSessionDirectory(const std::string& session_id) const {
    try {
        std::string session_dir = getSessionDirectory(session_id);
        std::filesystem::create_directories(session_dir);
        std::filesystem::create_directories(session_dir + "/checkpoints");
        std::filesystem::create_directories(session_dir + "/modules");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error creating session directory: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> LearningStateManager::getAvailableSessions() const {
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> sessions;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(base_save_path_)) {
            if (entry.is_directory()) {
                std::string session_id = entry.path().filename().string();
                if (session_id.rfind("session_", 0) == 0) {
                    auto ftime = std::filesystem::last_write_time(entry);
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                    sessions.emplace_back(session_id, sctp);
                }
            }
        }
        // Sort sessions by time
        std::sort(sessions.begin(), sessions.end(), 
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
    } catch (const std::exception& e) {
        std::cerr << "Error reading session directories: " << e.what() << std::endl;
    }
    return sessions;
}