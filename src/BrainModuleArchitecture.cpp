#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/LearningState.h"
// CUDA support is optional - forward declaration in header is sufficient
// #include "NeuroGen/cuda/NetworkCUDA.cuh"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <algorithm>

BrainModuleArchitecture::BrainModuleArchitecture(const ArchitectureConfig& config)
    : architecture_config_(config),
      modular_network_(std::make_unique<ModularNeuralNetwork>()),
      last_update_time_(std::chrono::steady_clock::now()),
      creation_time_(std::chrono::steady_clock::now()) {
    // Initialization logic here
}

BrainModuleArchitecture::~BrainModuleArchitecture() {
    // Cleanup logic here
}

bool BrainModuleArchitecture::initialize(int visual_input_width, int visual_input_height) {
    visual_input_width_ = visual_input_width;
    visual_input_height_ = visual_input_height;
    visual_feature_size_ = visual_input_width * visual_input_height;

    if (!initializeDefaultModules()) {
        return false;
    }
    
    initializeInterModuleConnections();
    return true;
}

std::shared_ptr<EnhancedNeuralModule> BrainModuleArchitecture::getModule(const std::string& module_name) const {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    auto it = modules_.find(module_name);
    if (it != modules_.end()) {
        return it->second;
    }
    return nullptr;
}

void BrainModuleArchitecture::update(float dt, float global_reward) {
    // Update logic here
}

// ============================================================================
// STATE PERSISTENCE METHODS
// ============================================================================

bool BrainModuleArchitecture::saveLearningState(const std::string& save_directory, const std::string& checkpoint_name) {
    // Create directory if it doesn't exist
    std::filesystem::create_directories(save_directory);
    
    try {
        // Get current global state
        SessionLearningState global_state = getGlobalLearningState();
        global_state.session_id = checkpoint_name;
        
        // Save global state
        std::string global_state_path = save_directory + "/" + checkpoint_name + "_global.learningstate";
        std::ofstream global_file(global_state_path, std::ios::binary);
        if (!global_file.is_open()) {
            std::cerr << "❌ Cannot open global state file: " << global_state_path << std::endl;
            return false;
        }
        
        // Write global state header
        std::string header = "NEUREGEN_GLOBAL_STATE_V1";
        global_file.write(header.c_str(), header.size());
        
        // Write global state data
        global_file.write(reinterpret_cast<const char*>(&global_state.total_learning_steps), sizeof(uint64_t));
        global_file.write(reinterpret_cast<const char*>(&global_state.cumulative_reward), sizeof(float));
        global_file.write(reinterpret_cast<const char*>(&global_state.total_modules), sizeof(uint32_t));
        global_file.write(reinterpret_cast<const char*>(&global_state.total_neurons), sizeof(uint32_t));
        global_file.write(reinterpret_cast<const char*>(&global_state.total_synapses), sizeof(uint32_t));
        global_file.write(reinterpret_cast<const char*>(&global_state.average_performance), sizeof(float));
        
        // Write architecture hash
        size_t hash_size = global_state.architecture_hash.size();
        global_file.write(reinterpret_cast<const char*>(&hash_size), sizeof(size_t));
        global_file.write(global_state.architecture_hash.c_str(), hash_size);
        
        // Write session metadata
        auto session_start_time = std::chrono::duration_cast<std::chrono::seconds>(
            global_state.session_start.time_since_epoch()).count();
        global_file.write(reinterpret_cast<const char*>(&session_start_time), sizeof(int64_t));
        
        global_file.close();
        
        // Save individual module states
        auto module_names = getModuleNames();
        bool all_modules_saved = true;
        
        for (const auto& module_name : module_names) {
            if (!saveModuleLearningState(module_name, save_directory)) {
                std::cerr << "⚠️  Failed to save learning state for module: " << module_name << std::endl;
                all_modules_saved = false;
            }
        }
        
        // Save inter-module connections
        std::string connections_path = save_directory + "/" + checkpoint_name + "_connections.state";
        std::ofstream connections_file(connections_path, std::ios::binary);
        if (connections_file.is_open()) {
            std::string conn_header = "NEUREGEN_CONNECTIONS_V1";
            connections_file.write(conn_header.c_str(), conn_header.size());
            
            size_t num_connections = inter_module_connections_.size();
            connections_file.write(reinterpret_cast<const char*>(&num_connections), sizeof(size_t));
            
            for (const auto& [connection_pair, connection_state] : inter_module_connections_) {
                // Write source module name
                size_t source_size = connection_pair.first.size();
                connections_file.write(reinterpret_cast<const char*>(&source_size), sizeof(size_t));
                connections_file.write(connection_pair.first.c_str(), source_size);
                
                // Write target module name
                size_t target_size = connection_pair.second.size();
                connections_file.write(reinterpret_cast<const char*>(&target_size), sizeof(size_t));
                connections_file.write(connection_pair.second.c_str(), target_size);
                
                // Write connection state data
                connections_file.write(reinterpret_cast<const char*>(&connection_state.connection_strength), sizeof(float));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.base_strength), sizeof(float));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.plasticity_rate), sizeof(float));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.usage_frequency), sizeof(float));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.correlation_strength), sizeof(float));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.activation_count), sizeof(uint64_t));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.information_transfer_rate), sizeof(float));
                connections_file.write(reinterpret_cast<const char*>(&connection_state.mutual_information), sizeof(float));
                
                // Write connection type
                size_t type_size = connection_state.connection_type.size();
                connections_file.write(reinterpret_cast<const char*>(&type_size), sizeof(size_t));
                connections_file.write(connection_state.connection_type.c_str(), type_size);
            }
            connections_file.close();
        }
        
        // Save attention weights
        std::string attention_path = save_directory + "/" + checkpoint_name + "_attention.state";
        std::ofstream attention_file(attention_path, std::ios::binary);
        if (attention_file.is_open()) {
            size_t num_modules = attention_weights_.size();
            attention_file.write(reinterpret_cast<const char*>(&num_modules), sizeof(size_t));
            
            for (const auto& [module_name, weight] : attention_weights_) {
                size_t name_size = module_name.size();
                attention_file.write(reinterpret_cast<const char*>(&name_size), sizeof(size_t));
                attention_file.write(module_name.c_str(), name_size);
                attention_file.write(reinterpret_cast<const char*>(&weight), sizeof(float));
            }
            attention_file.close();
        }
        
        std::cout << "✅ Saved learning state: " << checkpoint_name << std::endl;
        return all_modules_saved;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving learning state: " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::loadLearningState(const std::string& save_directory, const std::string& checkpoint_name) {
    try {
        std::string target_checkpoint = checkpoint_name;
        
        // If no specific checkpoint specified, find the latest
        if (target_checkpoint.empty()) {
            std::vector<std::string> available_checkpoints;
            
            for (const auto& entry : std::filesystem::directory_iterator(save_directory)) {
                if (entry.path().extension() == ".learningstate" && 
                    entry.path().filename().string().find("_global") != std::string::npos) {
                    std::string filename = entry.path().stem().string();
                    size_t pos = filename.find("_global");
                    if (pos != std::string::npos) {
                        available_checkpoints.push_back(filename.substr(0, pos));
                    }
                }
            }
            
            if (available_checkpoints.empty()) {
                std::cerr << "❌ No checkpoints found in directory: " << save_directory << std::endl;
                return false;
            }
            
            // Use the latest checkpoint (assume lexicographically sorted)
            std::sort(available_checkpoints.begin(), available_checkpoints.end());
            target_checkpoint = available_checkpoints.back();
        }
        
        // Load global state
        std::string global_state_path = save_directory + "/" + target_checkpoint + "_global.learningstate";
        std::ifstream global_file(global_state_path, std::ios::binary);
        if (!global_file.is_open()) {
            std::cerr << "❌ Cannot open global state file: " << global_state_path << std::endl;
            return false;
        }
        
        // Verify header
        std::string header = "NEUREGEN_GLOBAL_STATE_V1";
        std::vector<char> file_header(header.size());
        global_file.read(file_header.data(), header.size());
        if (std::string(file_header.begin(), file_header.end()) != header) {
            std::cerr << "❌ Invalid global state file format" << std::endl;
            return false;
        }
        
        // Read global state data
        uint64_t total_learning_steps;
        float cumulative_reward;
        uint32_t total_modules, total_neurons, total_synapses;
        float average_performance;
        
        global_file.read(reinterpret_cast<char*>(&total_learning_steps), sizeof(uint64_t));
        global_file.read(reinterpret_cast<char*>(&cumulative_reward), sizeof(float));
        global_file.read(reinterpret_cast<char*>(&total_modules), sizeof(uint32_t));
        global_file.read(reinterpret_cast<char*>(&total_neurons), sizeof(uint32_t));
        global_file.read(reinterpret_cast<char*>(&total_synapses), sizeof(uint32_t));
        global_file.read(reinterpret_cast<char*>(&average_performance), sizeof(float));
        
        // Read architecture hash
        size_t hash_size;
        global_file.read(reinterpret_cast<char*>(&hash_size), sizeof(size_t));
        std::vector<char> hash_buffer(hash_size);
        global_file.read(hash_buffer.data(), hash_size);
        std::string loaded_hash(hash_buffer.begin(), hash_buffer.end());
        
        // Validate architecture compatibility
        if (!validateArchitectureCompatibility(loaded_hash).first) {
            std::cerr << "⚠️  Architecture compatibility warning" << std::endl;
        }
        
        // Read session metadata
        int64_t session_start_time;
        global_file.read(reinterpret_cast<char*>(&session_start_time), sizeof(int64_t));
        
        global_file.close();
        
        // Apply loaded global state
        {
            std::lock_guard<std::mutex> lock(learning_state_mutex_);
            global_learning_steps_ = total_learning_steps;
            global_reward_accumulator_ = cumulative_reward;
        }
        
        // Load individual module states
        auto module_names = getModuleNames();
        bool all_modules_loaded = true;
        
        for (const auto& module_name : module_names) {
            if (!loadModuleLearningState(module_name, save_directory)) {
                std::cerr << "⚠️  Failed to load learning state for module: " << module_name << std::endl;
                all_modules_loaded = false;
            }
        }
        
        // Load inter-module connections
        std::string connections_path = save_directory + "/" + target_checkpoint + "_connections.state";
        std::ifstream connections_file(connections_path, std::ios::binary);
        if (connections_file.is_open()) {
            // Verify header
            std::string conn_header = "NEUREGEN_CONNECTIONS_V1";
            std::vector<char> conn_header_buffer(conn_header.size());
            connections_file.read(conn_header_buffer.data(), conn_header.size());
            
            if (std::string(conn_header_buffer.begin(), conn_header_buffer.end()) == conn_header) {
                size_t num_connections;
                connections_file.read(reinterpret_cast<char*>(&num_connections), sizeof(size_t));
                
                for (size_t i = 0; i < num_connections; ++i) {
                    // Read source module name
                    size_t source_size;
                    connections_file.read(reinterpret_cast<char*>(&source_size), sizeof(size_t));
                    std::vector<char> source_buffer(source_size);
                    connections_file.read(source_buffer.data(), source_size);
                    std::string source_name(source_buffer.begin(), source_buffer.end());
                    
                    // Read target module name
                    size_t target_size;
                    connections_file.read(reinterpret_cast<char*>(&target_size), sizeof(size_t));
                    std::vector<char> target_buffer(target_size);
                    connections_file.read(target_buffer.data(), target_size);
                    std::string target_name(target_buffer.begin(), target_buffer.end());
                    
                    // Find existing connection
                    std::pair<std::string, std::string> connection_key = {source_name, target_name};
                    if (inter_module_connections_.find(connection_key) != inter_module_connections_.end()) {
                        auto& connection_state = inter_module_connections_[connection_key];
                        
                        // Read connection state data
                        connections_file.read(reinterpret_cast<char*>(&connection_state.connection_strength), sizeof(float));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.base_strength), sizeof(float));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.plasticity_rate), sizeof(float));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.usage_frequency), sizeof(float));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.correlation_strength), sizeof(float));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.activation_count), sizeof(uint64_t));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.information_transfer_rate), sizeof(float));
                        connections_file.read(reinterpret_cast<char*>(&connection_state.mutual_information), sizeof(float));
                        
                        // Read connection type
                        size_t type_size;
                        connections_file.read(reinterpret_cast<char*>(&type_size), sizeof(size_t));
                        std::vector<char> type_buffer(type_size);
                        connections_file.read(type_buffer.data(), type_size);
                        connection_state.connection_type = std::string(type_buffer.begin(), type_buffer.end());
                    } else {
                        // Skip unknown connection
                        connections_file.seekg(7 * sizeof(float) + sizeof(uint64_t), std::ios::cur);
                        size_t type_size;
                        connections_file.read(reinterpret_cast<char*>(&type_size), sizeof(size_t));
                        connections_file.seekg(type_size, std::ios::cur);
                    }
                }
            }
            connections_file.close();
        }
        
        // Load attention weights
        std::string attention_path = save_directory + "/" + target_checkpoint + "_attention.state";
        std::ifstream attention_file(attention_path, std::ios::binary);
        if (attention_file.is_open()) {
            size_t num_modules;
            attention_file.read(reinterpret_cast<char*>(&num_modules), sizeof(size_t));
            
            for (size_t i = 0; i < num_modules; ++i) {
                size_t name_size;
                attention_file.read(reinterpret_cast<char*>(&name_size), sizeof(size_t));
                std::vector<char> name_buffer(name_size);
                attention_file.read(name_buffer.data(), name_size);
                std::string module_name(name_buffer.begin(), name_buffer.end());
                
                float weight;
                attention_file.read(reinterpret_cast<char*>(&weight), sizeof(float));
                
                if (attention_weights_.find(module_name) != attention_weights_.end()) {
                    attention_weights_[module_name] = weight;
                }
            }
            attention_file.close();
        }
        
        std::cout << "✅ Loaded learning state: " << target_checkpoint << std::endl;
        std::cout << "📊 Learning steps: " << total_learning_steps 
                  << ", Cumulative reward: " << cumulative_reward
                  << ", Avg performance: " << average_performance << std::endl;
        
        return all_modules_loaded;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading learning state: " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::saveModuleLearningState(const std::string& module_name, const std::string& save_directory) const {
    auto module = getModule(module_name);
    if (!module) {
        return false;
    }
    
    std::string module_state_path = save_directory + "/" + module_name + "_module.state";
    return module->save_state(module_state_path);
}

bool BrainModuleArchitecture::loadModuleLearningState(const std::string& module_name, const std::string& save_directory) {
    auto module = getModule(module_name);
    if (!module) {
        return false;
    }
    
    std::string module_state_path = save_directory + "/" + module_name + "_module.state";
    return module->load_state(module_state_path);
}

// ============================================================================
// INTER-MODULE CONNECTION STATE METHODS
// ============================================================================

std::vector<InterModuleConnectionState> BrainModuleArchitecture::getInterModuleConnectionState() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    std::vector<InterModuleConnectionState> connections;
    connections.reserve(inter_module_connections_.size());
    
    for (const auto& [connection_pair, connection_state] : inter_module_connections_) {
        connections.push_back(connection_state);
    }
    
    return connections;
}

bool BrainModuleArchitecture::applyInterModuleConnectionState(const std::vector<InterModuleConnectionState>& connections) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    try {
        for (const auto& connection_state : connections) {
            std::pair<std::string, std::string> connection_key = {
                connection_state.source_module, 
                connection_state.target_module
            };
            
            if (inter_module_connections_.find(connection_key) != inter_module_connections_.end()) {
                inter_module_connections_[connection_key] = connection_state;
            } else {
                std::cerr << "⚠️  Connection not found: " << connection_state.source_module 
                          << " -> " << connection_state.target_module << std::endl;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error applying inter-module connection state: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// PERFORMANCE AND MONITORING METHODS
// ============================================================================

std::map<std::string, float> BrainModuleArchitecture::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    std::map<std::string, float> metrics;
    
    // Global metrics
    metrics["total_modules"] = static_cast<float>(modules_.size());
    metrics["total_connections"] = static_cast<float>(inter_module_connections_.size());
    metrics["global_learning_steps"] = static_cast<float>(global_learning_steps_);
    metrics["cumulative_reward"] = global_reward_accumulator_;
    metrics["total_activity"] = getTotalActivity();
    
    // Module-specific metrics
    for (const auto& [module_name, history] : module_performance_history_) {
        if (!history.empty()) {
            float avg_performance = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
            metrics["module_" + module_name + "_avg_performance"] = avg_performance;
            
            // Recent performance (last 100 steps)
            size_t recent_count = std::min(history.size(), size_t(100));
            float recent_avg = std::accumulate(history.end() - recent_count, history.end(), 0.0f) / recent_count;
            metrics["module_" + module_name + "_recent_performance"] = recent_avg;
        }
        
        if (module_prediction_errors_.find(module_name) != module_prediction_errors_.end()) {
            metrics["module_" + module_name + "_prediction_error"] = module_prediction_errors_.at(module_name);
        }
        
        if (attention_weights_.find(module_name) != attention_weights_.end()) {
            metrics["module_" + module_name + "_attention_weight"] = attention_weights_.at(module_name);
        }
    }
    
    // Connection metrics
    float avg_connection_strength = 0.0f;
    float avg_usage_frequency = 0.0f;
    if (!inter_module_connections_.empty()) {
        for (const auto& [_, connection_state] : inter_module_connections_) {
            avg_connection_strength += connection_state.connection_strength;
            avg_usage_frequency += connection_state.usage_frequency;
        }
        avg_connection_strength /= inter_module_connections_.size();
        avg_usage_frequency /= inter_module_connections_.size();
    }
    
    metrics["avg_connection_strength"] = avg_connection_strength;
    metrics["avg_connection_usage"] = avg_usage_frequency;
    
    return metrics;
}

float BrainModuleArchitecture::getTotalActivity() const {
    float total_activity = 0.0f;
    size_t module_count = 0;
    
    auto module_names = getModuleNames();
    for (const auto& module_name : module_names) {
        auto module = getModule(module_name);
        if (module) {
            total_activity += module->getActivityLevel();
            module_count++;
        }
    }
    
    return module_count > 0 ? total_activity / module_count : 0.0f;
}

std::pair<bool, std::string> BrainModuleArchitecture::isStable() const {
    // Check if all modules have stable activity levels
    auto module_names = getModuleNames();
    
    for (const auto& module_name : module_names) {
        auto module = getModule(module_name);
        if (module) {
            float activity = module->getActivityLevel();
            
            // Check for runaway activity or complete silence
            if (activity > 2.0f || activity < 0.001f) {
                return {false, "Module " + module_name + " has unstable activity: " + std::to_string(activity)};
            }
        }
    }
    
    // Check connection stability
    for (const auto& [_, connection_state] : inter_module_connections_) {
        if (connection_state.connection_strength > 1.5f || connection_state.connection_strength < 0.0f) {
            return {false, "Inter-module connection has unstable strength: " + std::to_string(connection_state.connection_strength)};
        }
    }
    
    return {true, "Architecture is stable"};
}

std::pair<bool, std::string> BrainModuleArchitecture::validateArchitectureCompatibility(const std::string& state_hash) const {
    std::string current_hash = calculateArchitectureHash();
    
    if (current_hash == state_hash) {
        return {true, "Architecture hash matches."};
    }
    
    // Allow minor differences (could implement fuzzy matching here)
    std::string message = "Architecture hash mismatch:\n    Current:  " + current_hash + "\n    Expected: " + state_hash;
    std::cout << "⚠️  " << message << std::endl;
    
    return {false, message}; // Strict validation
}

void BrainModuleArchitecture::updateConnectionUsage(const std::string& source_module, 
                                                   const std::string& target_module, 
                                                   float activation_strength) {
    std::pair<std::string, std::string> connection_key = {source_module, target_module};
    
    if (connection_usage_history_.find(connection_key) != connection_usage_history_.end()) {
        // Update usage with exponential moving average
        connection_usage_history_[connection_key] = 
            0.9f * connection_usage_history_[connection_key] + 0.1f * activation_strength;
    }
}

// ============================================================================
// GPU INTEGRATION METHODS
// ============================================================================

void BrainModuleArchitecture::setCUDANetwork(std::shared_ptr<NetworkCUDA> cuda_network) {
    cuda_network_ = cuda_network;

    if (cuda_network_) {
        // CUDA functionality disabled in this build
        // cuda_network_->setBrainArchitecture(shared_from_this());
        std::cout << "⚠️  CUDA support not available in this build" << std::endl;
    }
}

std::pair<bool, std::string> BrainModuleArchitecture::enableGPUAcceleration(bool enable) {
    if (enable && !cuda_network_) {
        std::string msg = "⚠️  CUDA network not set - cannot enable GPU acceleration";
        std::cerr << msg << std::endl;
        return {false, msg};
    }
    
    gpu_enabled_ = enable;
    
    if (enable) {
        std::string msg = "🚀 GPU acceleration enabled";
        std::cout << msg << std::endl;
        return {true, msg};
    } else {
        std::string msg = "🔄 GPU acceleration disabled";
        std::cout << msg << std::endl;
        return {true, msg};
    }
}

bool BrainModuleArchitecture::isGPUEnabled() const {
    return gpu_enabled_ && cuda_network_ != nullptr;
}

// ============================================================================
// INITIALIZATION HELPER METHODS
// ============================================================================

bool BrainModuleArchitecture::initializeDefaultModules() {
    // Implement default module initialization
    // This method should initialize the basic modules needed for the brain architecture
    
    return true;
}

void BrainModuleArchitecture::initializeInterModuleConnections() {
    // Implement inter-module connection initialization
    // This method should set up the connections between modules
}

// ============================================================================
// PLACEHOLDER METHODS FOR UNDEFINED BEHAVIOR
// ============================================================================

SessionLearningState BrainModuleArchitecture::getGlobalLearningState() const {
    // TODO: Implement global learning state retrieval
    return SessionLearningState{};
}

std::vector<std::string> BrainModuleArchitecture::getModuleNames() const {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    std::vector<std::string> names;
    for (const auto& pair : modules_) {
        names.push_back(pair.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::string BrainModuleArchitecture::calculateArchitectureHash() const {
    // TODO: Implement a proper hash calculation
    return "dummy_hash";
}

std::map<std::string, uint32_t> BrainModuleArchitecture::getArchitectureStatistics() const {
    // TODO: Implement architecture statistics retrieval
    return {};
}

size_t BrainModuleArchitecture::performGlobalMemoryConsolidation(float consolidation_strength) {
    // TODO: Implement global memory consolidation
    return 0;
}

size_t BrainModuleArchitecture::getModuleCount() const {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    return modules_.size();
}