#include <NeuroGen/AutonomousLearningAgent.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

// ============================================================================
// MEMORY SYSTEM IMPLEMENTATION
// ============================================================================

MemorySystem::MemorySystem(size_t max_episodes, size_t working_capacity)
    : max_episodes_per_cluster_(1000),
      max_total_episodes_(max_episodes),
      consolidation_threshold_(0.7f),
      forgetting_rate_(0.001f),
      working_memory_capacity_(working_capacity) {
    
    working_memory_.reserve(working_memory_capacity_);
    std::cout << "MemorySystem: Initialized with capacity " << max_episodes << std::endl;
}

void MemorySystem::storeEpisode(const MemoryTrace& episode, const std::string& context) {
    // Find or create the appropriate episodic cluster
    if (episodic_memory_.find(context) == episodic_memory_.end()) {
        EpisodicCluster new_cluster;
        new_cluster.context_type = context;
        new_cluster.cluster_coherence = 1.0f;
        new_cluster.prototype_state = episode.state_vector;
        new_cluster.access_count = 0;
        new_cluster.last_accessed = std::chrono::steady_clock::now();
        episodic_memory_[context] = new_cluster;
    }
    
    EpisodicCluster& cluster = episodic_memory_[context];
    
    // Add episode to cluster
    cluster.episodes.push_back(episode);
    cluster.access_count++;
    cluster.last_accessed = std::chrono::steady_clock::now();
    
    // Update prototype state (running average)
    if (episode.state_vector.size() == cluster.prototype_state.size()) {
        float alpha = 0.1f;  // Learning rate for prototype update
        for (size_t i = 0; i < cluster.prototype_state.size(); ++i) {
            cluster.prototype_state[i] = cluster.prototype_state[i] * (1.0f - alpha) + 
                                        episode.state_vector[i] * alpha;
        }
    }
    
    // Maintain cluster size limit
    if (cluster.episodes.size() > max_episodes_per_cluster_) {
        // Remove oldest episodes or least important ones
        std::sort(cluster.episodes.begin(), cluster.episodes.end(),
                  [](const MemoryTrace& a, const MemoryTrace& b) {
                      return a.importance_weight > b.importance_weight;
                  });
        cluster.episodes.resize(max_episodes_per_cluster_);
    }
    
    // Update cluster coherence
    updateClusterCoherence(cluster);
    
    std::cout << "MemorySystem: Stored episode in context '" << context 
              << "' (total: " << cluster.episodes.size() << ")" << std::endl;
}

std::vector<MemorySystem::MemoryTrace> MemorySystem::retrieveSimilarEpisodes(
    const std::vector<float>& current_state, 
    const std::string& context,
    size_t max_results) {
    
    std::vector<MemoryTrace> similar_episodes;
    similar_episodes.reserve(max_results * 2);  // Reserve extra space
    
    // If context is specified, search only in that cluster
    if (!context.empty() && episodic_memory_.find(context) != episodic_memory_.end()) {
        EpisodicCluster& cluster = episodic_memory_[context];
        searchClusterForSimilarEpisodes(cluster, current_state, similar_episodes);
        cluster.access_count++;
        cluster.last_accessed = std::chrono::steady_clock::now();
    } else {
        // Search across all clusters
        for (auto& [ctx, cluster] : episodic_memory_) {
            searchClusterForSimilarEpisodes(cluster, current_state, similar_episodes);
        }
    }
    
    // Sort by similarity (stored in importance_weight for this function)
    std::sort(similar_episodes.begin(), similar_episodes.end(),
              [](const MemoryTrace& a, const MemoryTrace& b) {
                  return a.importance_weight > b.importance_weight;
              });
    
    // Limit results
    if (similar_episodes.size() > max_results) {
        similar_episodes.resize(max_results);
    }
    
    return similar_episodes;
}

void MemorySystem::consolidateMemories() {
    std::cout << "MemorySystem: Starting memory consolidation..." << std::endl;
    
    for (auto& [context, cluster] : episodic_memory_) {
        // Mark important episodes for consolidation
        for (MemoryTrace& episode : cluster.episodes) {
            if (!episode.is_consolidated && 
                episode.importance_weight > consolidation_threshold_) {
                
                episode.is_consolidated = true;
                episode.importance_weight *= 1.1f;  // Boost consolidated memories
                
                // Update skill memory based on successful episodes
                if (episode.reward_received > 0.5f) {
                    updateSkillLevel(context, episode.reward_received * 0.1f);
                }
            }
        }
        
        // Update cluster coherence after consolidation
        updateClusterCoherence(cluster);
    }
    
    // Forget old, unimportant episodes
    forgetOldEpisodes();
    
    std::cout << "MemorySystem: Memory consolidation complete" << std::endl;
}

void MemorySystem::updateWorkingMemory(const MemoryTrace& trace) {
    working_memory_.push_back(trace);
    
    // Maintain working memory capacity
    if (working_memory_.size() > working_memory_capacity_) {
        // Remove oldest entry (FIFO)
        working_memory_.erase(working_memory_.begin());
    }
}

void MemorySystem::forgetOldEpisodes() {
    auto now = std::chrono::steady_clock::now();
    
    for (auto& [context, cluster] : episodic_memory_) {
        cluster.episodes.erase(
            std::remove_if(cluster.episodes.begin(), cluster.episodes.end(),
                          [&](const MemoryTrace& episode) {
                              // Calculate age in hours
                              auto age = std::chrono::duration_cast<std::chrono::hours>(
                                  now - episode.timestamp).count();
                              
                              // Forget if old, unimportant, and not consolidated
                              return (age > 24 &&  // Older than 24 hours
                                     episode.importance_weight < 0.3f &&
                                     !episode.is_consolidated);
                          }),
            cluster.episodes.end());
    }
}

void MemorySystem::strengthenMemory(const std::string& context, int episode_id, float strength_boost) {
    auto it = episodic_memory_.find(context);
    if (it != episodic_memory_.end()) {
        EpisodicCluster& cluster = it->second;
        
        for (MemoryTrace& episode : cluster.episodes) {
            if (episode.episode_id == episode_id) {
                episode.importance_weight += strength_boost;
                episode.importance_weight = std::min(episode.importance_weight, 2.0f);  // Cap at 2.0
                std::cout << "MemorySystem: Strengthened memory " << episode_id 
                          << " in context '" << context << "'" << std::endl;
                break;
            }
        }
    }
}

float MemorySystem::getSkillLevel(const std::string& skill_name) const {
    auto it = skill_memory_.find(skill_name);
    return (it != skill_memory_.end()) ? it->second : 0.0f;
}

void MemorySystem::updateSkillLevel(const std::string& skill_name, float performance) {
    float current_level = getSkillLevel(skill_name);
    float learning_rate = 0.1f;
    
    // Update skill level with exponential moving average
    skill_memory_[skill_name] = current_level * (1.0f - learning_rate) + 
                               performance * learning_rate;
    
    // Clamp to [0, 1] range
    skill_memory_[skill_name] = std::clamp(skill_memory_[skill_name], 0.0f, 1.0f);
}

std::vector<std::string> MemorySystem::getKnownContexts() const {
    std::vector<std::string> contexts;
    contexts.reserve(episodic_memory_.size());
    
    for (const auto& [context, cluster] : episodic_memory_) {
        contexts.push_back(context);
    }
    
    return contexts;
}

size_t MemorySystem::getEpisodeCount(const std::string& context) const {
    if (context.empty()) {
        // Return total count across all contexts
        size_t total = 0;
        for (const auto& [ctx, cluster] : episodic_memory_) {
            total += cluster.episodes.size();
        }
        return total;
    } else {
        auto it = episodic_memory_.find(context);
        return (it != episodic_memory_.end()) ? it->second.episodes.size() : 0;
    }
}

bool MemorySystem::saveMemoryState(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "MemorySystem: Cannot open file for saving: " << filename << std::endl;
        return false;
    }
    
    try {
        // Write header
        const char* header = "NEUROGENMEMORY";
        file.write(header, 14);
        
        // Write version
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write number of contexts
        uint32_t num_contexts = static_cast<uint32_t>(episodic_memory_.size());
        file.write(reinterpret_cast<const char*>(&num_contexts), sizeof(num_contexts));
        
        // Write each context and its episodes
        for (const auto& [context, cluster] : episodic_memory_) {
            // Write context name length and name
            uint32_t context_len = static_cast<uint32_t>(context.length());
            file.write(reinterpret_cast<const char*>(&context_len), sizeof(context_len));
            file.write(context.c_str(), context_len);
            
            // Write number of episodes in this cluster
            uint32_t num_episodes = static_cast<uint32_t>(cluster.episodes.size());
            file.write(reinterpret_cast<const char*>(&num_episodes), sizeof(num_episodes));
            
            // Write each episode (simplified - only key data)
            for (const MemoryTrace& episode : cluster.episodes) {
                // Write state vector size and data
                uint32_t state_size = static_cast<uint32_t>(episode.state_vector.size());
                file.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
                if (state_size > 0) {
                    file.write(reinterpret_cast<const char*>(episode.state_vector.data()),
                              state_size * sizeof(float));
                }
                
                // Write scalar values
                file.write(reinterpret_cast<const char*>(&episode.reward_received), sizeof(float));
                file.write(reinterpret_cast<const char*>(&episode.importance_weight), sizeof(float));
                file.write(reinterpret_cast<const char*>(&episode.is_consolidated), sizeof(bool));
            }
        }
        
        // Write skill memory
        uint32_t num_skills = static_cast<uint32_t>(skill_memory_.size());
        file.write(reinterpret_cast<const char*>(&num_skills), sizeof(num_skills));
        
        for (const auto& [skill, level] : skill_memory_) {
            uint32_t skill_len = static_cast<uint32_t>(skill.length());
            file.write(reinterpret_cast<const char*>(&skill_len), sizeof(skill_len));
            file.write(skill.c_str(), skill_len);
            file.write(reinterpret_cast<const char*>(&level), sizeof(float));
        }
        
        std::cout << "MemorySystem: Saved memory state to " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "MemorySystem: Error saving memory state: " << e.what() << std::endl;
        return false;
    }
}

bool MemorySystem::loadMemoryState(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "MemorySystem: Cannot open file for loading: " << filename << std::endl;
        return false;
    }
    
    try {
        // Verify header
        char header[15] = {0};
        file.read(header, 14);
        if (std::string(header) != "NEUROGENMEMORY") {
            std::cerr << "MemorySystem: Invalid file format" << std::endl;
            return false;
        }
        
        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            std::cerr << "MemorySystem: Unsupported version " << version << std::endl;
            return false;
        }
        
        // Clear existing memory
        episodic_memory_.clear();
        skill_memory_.clear();
        
        // Read contexts and episodes
        uint32_t num_contexts;
        file.read(reinterpret_cast<char*>(&num_contexts), sizeof(num_contexts));
        
        for (uint32_t i = 0; i < num_contexts; ++i) {
            // Read context name
            uint32_t context_len;
            file.read(reinterpret_cast<char*>(&context_len), sizeof(context_len));
            
            std::string context(context_len, '\0');
            file.read(&context[0], context_len);
            
            // Create new cluster
            EpisodicCluster cluster;
            cluster.context_type = context;
            cluster.cluster_coherence = 1.0f;
            cluster.access_count = 0;
            cluster.last_accessed = std::chrono::steady_clock::now();
            
            // Read episodes
            uint32_t num_episodes;
            file.read(reinterpret_cast<char*>(&num_episodes), sizeof(num_episodes));
            
            for (uint32_t j = 0; j < num_episodes; ++j) {
                MemoryTrace episode;
                
                // Read state vector
                uint32_t state_size;
                file.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
                
                episode.state_vector.resize(state_size);
                if (state_size > 0) {
                    file.read(reinterpret_cast<char*>(episode.state_vector.data()),
                             state_size * sizeof(float));
                }
                
                // Read scalar values
                file.read(reinterpret_cast<char*>(&episode.reward_received), sizeof(float));
                file.read(reinterpret_cast<char*>(&episode.importance_weight), sizeof(float));
                file.read(reinterpret_cast<char*>(&episode.is_consolidated), sizeof(bool));
                
                // Set other fields to defaults
                episode.episode_id = static_cast<int>(j);
                episode.timestamp = std::chrono::steady_clock::now();
                episode.episode_context = context;
                
                cluster.episodes.push_back(episode);
            }
            
            // Update prototype state if episodes exist
            if (!cluster.episodes.empty() && !cluster.episodes[0].state_vector.empty()) {
                cluster.prototype_state = cluster.episodes[0].state_vector;
            }
            
            episodic_memory_[context] = cluster;
        }
        
        // Read skill memory
        uint32_t num_skills;
        file.read(reinterpret_cast<char*>(&num_skills), sizeof(num_skills));
        
        for (uint32_t i = 0; i < num_skills; ++i) {
            uint32_t skill_len;
            file.read(reinterpret_cast<char*>(&skill_len), sizeof(skill_len));
            
            std::string skill(skill_len, '\0');
            file.read(&skill[0], skill_len);
            
            float level;
            file.read(reinterpret_cast<char*>(&level), sizeof(float));
            
            skill_memory_[skill] = level;
        }
        
        std::cout << "MemorySystem: Loaded memory state from " << filename << std::endl;
        std::cout << "  Contexts: " << episodic_memory_.size() << std::endl;
        std::cout << "  Skills: " << skill_memory_.size() << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "MemorySystem: Error loading memory state: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// PRIVATE HELPER METHODS
// ============================================================================

void MemorySystem::searchClusterForSimilarEpisodes(
    const EpisodicCluster& cluster,
    const std::vector<float>& current_state,
    std::vector<MemoryTrace>& results) {
    
    for (const MemoryTrace& episode : cluster.episodes) {
        if (episode.state_vector.size() != current_state.size()) {
            continue;  // Skip episodes with incompatible state dimensions
        }
        
        // Compute cosine similarity
        float similarity = computeCosineSimilarity(episode.state_vector, current_state);
        
        // Only include episodes above similarity threshold
        if (similarity > 0.5f) {
            MemoryTrace similar_episode = episode;
            similar_episode.importance_weight = similarity;  // Store similarity temporarily
            results.push_back(similar_episode);
        }
    }
}

float MemorySystem::computeCosineSimilarity(const std::vector<float>& vec1, 
                                           const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) return 0.0f;
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    float magnitude_product = std::sqrt(norm1) * std::sqrt(norm2);
    return (magnitude_product > 0.0f) ? (dot_product / magnitude_product) : 0.0f;
}

void MemorySystem::updateClusterCoherence(EpisodicCluster& cluster) {
    if (cluster.episodes.size() < 2) {
        cluster.cluster_coherence = 1.0f;
        return;
    }
    
    // Compute average pairwise similarity within the cluster
    float total_similarity = 0.0f;
    int comparisons = 0;
    
    for (size_t i = 0; i < cluster.episodes.size(); ++i) {
        for (size_t j = i + 1; j < cluster.episodes.size(); ++j) {
            float similarity = computeCosineSimilarity(
                cluster.episodes[i].state_vector,
                cluster.episodes[j].state_vector);
            total_similarity += similarity;
            comparisons++;
        }
    }
    
    cluster.cluster_coherence = (comparisons > 0) ? (total_similarity / comparisons) : 1.0f;
}

void MemorySystem::organizeMemoryStructure() {
    std::cout << "MemorySystem: Organizing memory structure..." << std::endl;
    
    // Remove empty clusters
    for (auto it = episodic_memory_.begin(); it != episodic_memory_.end();) {
        if (it->second.episodes.empty()) {
            std::cout << "  Removing empty cluster: " << it->first << std::endl;
            it = episodic_memory_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Update all cluster coherences
    for (auto& [context, cluster] : episodic_memory_) {
        updateClusterCoherence(cluster);
        
        // Sort episodes by importance within each cluster
        std::sort(cluster.episodes.begin(), cluster.episodes.end(),
                  [](const MemoryTrace& a, const MemoryTrace& b) {
                      return a.importance_weight > b.importance_weight;
                  });
    }
    
    std::cout << "MemorySystem: Memory organization complete" << std::endl;
}

// ============================================================================
// ADDITIONAL COMPATIBILITY METHODS
// ============================================================================

std::vector<float> MemorySystem::get_working_memory() const {
    return current_working_memory_;
}

void MemorySystem::update_working_memory(const std::vector<float>& new_memory) {
    current_working_memory_ = new_memory;
    
    // Also update working memory traces for compatibility
    if (new_memory.size() > 0) {
        MemoryTrace trace;
        trace.state_vector = new_memory;
        trace.timestamp = std::chrono::steady_clock::now();
        updateWorkingMemory(trace);
    }
}

void MemorySystem::store_episode(const std::vector<float>& state, 
                                 const std::vector<float>& action, 
                                 float reward, 
                                 float importance) {
    MemoryTrace episode;
    episode.state_vector = state;
    episode.action_vector = action;
    episode.reward = reward;
    episode.reward_received = reward;
    episode.importance_weight = importance;
    episode.timestamp = std::chrono::steady_clock::now();
    episode.is_consolidated = false;
    
    storeEpisode(episode, "default");
}

std::vector<MemorySystem::MemoryTrace> MemorySystem::retrieve_similar_episodes(
    const std::vector<float>& current_state, 
    size_t max_results) {
    return retrieveSimilarEpisodes(current_state, "", max_results);
}