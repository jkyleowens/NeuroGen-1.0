#ifndef MEMORY_SYSTEM_H
#define MEMORY_SYSTEM_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <cstdint>

/**
 * @brief Memory system for autonomous learning agent
 */
class MemorySystem {
public:
    /**
     * @brief Memory trace structure for episodic memory
     */
    struct MemoryTrace {
        std::vector<float> state_vector;
        std::vector<float> action_vector;
        float reward_received = 0.0f;
        float reward = 0.0f;  // Alias for backward compatibility
        float importance_weight = 0.5f;
        float temporal_discount = 0.99f;
        bool is_consolidated = false;
        std::chrono::steady_clock::time_point timestamp;
        std::string episode_context;
        std::string context_description;  // Alias for backward compatibility
        int episode_id = -1;
        
        MemoryTrace() {
            timestamp = std::chrono::steady_clock::now();
        }
    };
    
    /**
     * @brief Episodic memory cluster structure
     */
    struct EpisodicCluster {
        std::vector<MemoryTrace> episodes;
        std::vector<float> prototype_state;
        std::string context_type;
        float cluster_coherence = 1.0f;
        uint32_t access_count = 0;
        std::chrono::steady_clock::time_point last_accessed;
    };
    
    // Constructor
    MemorySystem(size_t max_episodes = 10000, size_t working_capacity = 100);
    
    // Core memory operations
    void storeEpisode(const MemoryTrace& episode, const std::string& context = "default");
    std::vector<MemoryTrace> retrieveSimilarEpisodes(const std::vector<float>& current_state, 
                                                      const std::string& context = "",
                                                      size_t max_results = 10);
    void consolidateMemories();
    void updateWorkingMemory(const MemoryTrace& trace);
    
    // Working memory operations
    std::vector<float> get_working_memory() const;
    void update_working_memory(const std::vector<float>& new_memory);
    
    // Memory management
    void forgetOldEpisodes();
    void strengthenMemory(const std::string& context, int episode_id, float strength_boost);
    
    // Skill and procedural memory
    float getSkillLevel(const std::string& skill_name) const;
    void updateSkillLevel(const std::string& skill_name, float performance);
    
    // Memory statistics and introspection
    std::vector<std::string> getKnownContexts() const;
    size_t getEpisodeCount(const std::string& context = "") const;
    
    // Persistence
    bool saveMemoryState(const std::string& filename) const;
    bool loadMemoryState(const std::string& filename);
    
    // Memory similarity functions
    std::vector<MemoryTrace> retrieve_similar_episodes(const std::vector<float>& state, size_t max_results);
    void store_episode(const std::vector<float>& state, const std::vector<float>& action, 
                      float reward, float confidence);
    
    // Public access to episodic memory for debugging/introspection
    const std::unordered_map<std::string, EpisodicCluster>& get_episodic_memory() const { return episodic_memory_; }
    size_t get_episodic_memory_size() const { return episodic_memory_.size(); }

private:
    // Memory storage
    std::unordered_map<std::string, EpisodicCluster> episodic_memory_;
    std::unordered_map<std::string, float> skill_memory_;
    std::vector<MemoryTrace> working_memory_;
    std::vector<float> current_working_memory_;
    
    // Configuration
    size_t max_episodes_per_cluster_;
    size_t max_total_episodes_;
    size_t working_memory_capacity_;
    float consolidation_threshold_;
    float forgetting_rate_;
    
    // Helper methods
    void searchClusterForSimilarEpisodes(const EpisodicCluster& cluster,
                                       const std::vector<float>& current_state,
                                       std::vector<MemoryTrace>& results);
    void updateClusterCoherence(EpisodicCluster& cluster);
    void organizeMemoryStructure();
    float computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};

#endif // MEMORY_SYSTEM_H
