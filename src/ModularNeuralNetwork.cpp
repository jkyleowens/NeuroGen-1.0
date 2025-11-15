#include <NeuroGen/ModularNeuralNetwork.h>
#include <NeuroGen/NeuralModule.h>
#include <iostream>
#include <algorithm>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

ModularNeuralNetwork::ModularNeuralNetwork()
    : is_initialized_(false),
      is_shutdown_(false),
      total_activity_(0.0f),
      update_count_(0) {
}

ModularNeuralNetwork::~ModularNeuralNetwork() {
    shutdown();
}

bool ModularNeuralNetwork::initialize() {
    if (is_initialized_) {
        return true;
    }
    
    // Initialize all registered modules
    for (auto& module : modules_) {
        if (module && !module->initialize()) {
            std::cerr << "ModularNeuralNetwork: Failed to initialize module " 
                      << module->get_name() << std::endl;
            return false;
        }
    }
    
    is_initialized_ = true;
    is_shutdown_ = false;
    
    std::cout << "ModularNeuralNetwork: Initialized with " 
              << modules_.size() << " modules" << std::endl;
    
    return true;
}

void ModularNeuralNetwork::shutdown() {
    if (is_shutdown_) {
        return;
    }
    
    // Clear module references
    module_map_.clear();
    modules_.clear();
    
    is_shutdown_ = true;
    is_initialized_ = false;
    
    std::cout << "ModularNeuralNetwork: Shutdown complete" << std::endl;
}

// ============================================================================
// MODULE MANAGEMENT
// ============================================================================

void ModularNeuralNetwork::add_module(std::unique_ptr<NeuralModule> module) {
    if (!module) {
        std::cerr << "ModularNeuralNetwork: Cannot add null module" << std::endl;
        return;
    }
    
    std::string module_name = module->get_name();
    
    // Check for duplicate names
    if (module_map_.find(module_name) != module_map_.end()) {
        std::cerr << "ModularNeuralNetwork: Module with name '" 
                  << module_name << "' already exists" << std::endl;
        return;
    }
    
    // Add to map for quick lookup
    module_map_[module_name] = module.get();
    
    // Add to storage
    modules_.push_back(std::move(module));
    
    std::cout << "ModularNeuralNetwork: Added module '" << module_name << "'" << std::endl;
}

bool ModularNeuralNetwork::remove_module(const std::string& module_name) {
    // Remove from map
    auto map_it = module_map_.find(module_name);
    if (map_it == module_map_.end()) {
        return false;
    }
    
    NeuralModule* module_ptr = map_it->second;
    module_map_.erase(map_it);
    
    // Remove from vector
    auto vec_it = std::find_if(modules_.begin(), modules_.end(),
        [module_ptr](const std::unique_ptr<NeuralModule>& m) {
            return m.get() == module_ptr;
        });
    
    if (vec_it != modules_.end()) {
        modules_.erase(vec_it);
        std::cout << "ModularNeuralNetwork: Removed module '" << module_name << "'" << std::endl;
        return true;
    }
    
    return false;
}

NeuralModule* ModularNeuralNetwork::get_module(const std::string& module_name) const {
    auto it = module_map_.find(module_name);
    return (it != module_map_.end()) ? it->second : nullptr;
}

std::vector<std::string> ModularNeuralNetwork::get_module_names() const {
    std::vector<std::string> names;
    names.reserve(module_map_.size());
    
    for (const auto& [name, module] : module_map_) {
        names.push_back(name);
    }
    
    return names;
}

size_t ModularNeuralNetwork::get_module_count() const {
    return modules_.size();
}

// ============================================================================
// NETWORK OPERATIONS
// ============================================================================

void ModularNeuralNetwork::update(float dt) {
    if (!is_initialized_ || is_shutdown_) {
        return;
    }
    
    total_activity_ = 0.0f;
    size_t active_modules = 0;
    
    // Update all active modules
    for (auto& module : modules_) {
        if (module && module->is_active()) {
            try {
                module->update(dt);
                
                // Accumulate activity
                auto output = module->get_output();
                float module_activity = 0.0f;
                for (float value : output) {
                    module_activity += std::abs(value);
                }
                
                total_activity_ += module_activity;
                active_modules++;
                
            } catch (const std::exception& e) {
                std::cerr << "ModularNeuralNetwork: Exception updating module " 
                          << module->get_name() << ": " << e.what() << std::endl;
            }
        }
    }
    
    // Calculate average activity
    if (active_modules > 0) {
        total_activity_ /= active_modules;
    }
    
    update_count_++;
    
    // Process inter-module communication
    process_inter_module_communication();
    
    // Update performance metrics periodically
    if (update_count_ % 1000 == 0) {
        update_performance_metrics();
    }
}

void ModularNeuralNetwork::update_module(const std::string& module_name, float dt) {
    auto module = get_module(module_name);
    if (module && module->is_active()) {
        module->update(dt);
    }
}

void ModularNeuralNetwork::process_inter_module_communication() {
    // Simple inter-module communication processing
    // In a more sophisticated implementation, this would handle:
    // - Signal routing between modules
    // - Priority-based message passing
    // - Temporal delays and buffering
    
    // For now, just ensure all modules have processed their communications
    for (auto& module : modules_) {
        if (module && module->is_active()) {
            // If the module has processInterModuleCommunication method, call it
            // This would be module-specific processing
        }
    }
}

void ModularNeuralNetwork::reset() {
    total_activity_ = 0.0f;
    update_count_ = 0;
    
    // Reset all modules
    for (auto& module : modules_) {
        if (module) {
            // Reset module state - this would depend on the module interface
            // For now, just reactivate if needed
            if (!module->is_active()) {
                module->set_active(true);
            }
        }
    }
    
    std::cout << "ModularNeuralNetwork: Reset complete" << std::endl;
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

std::map<std::string, std::map<std::string, float>> ModularNeuralNetwork::getPerformanceMetrics() const {
    std::map<std::string, std::map<std::string, float>> all_metrics;
    
    for (const auto& [name, module] : module_map_) {
        if (module) {
            all_metrics[name] = module->getPerformanceMetrics();
        }
    }
    
    return all_metrics;
}

float ModularNeuralNetwork::get_total_activity() const {
    return total_activity_;
}

bool ModularNeuralNetwork::is_stable() const {
    if (!is_initialized_ || is_shutdown_) {
        return false;
    }
    
    // Check if all modules are operating within normal parameters
    for (const auto& [name, module] : module_map_) {
        if (module) {
            auto metrics = module->getPerformanceMetrics();
            
            // Simple stability checks
            auto activity_it = metrics.find("average_activity");
            if (activity_it != metrics.end()) {
                if (activity_it->second > 10.0f || activity_it->second < 0.0f) {
                    return false;  // Activity out of normal range
                }
            }
        }
    }
    
    return true;
}

// ============================================================================
// PRIVATE HELPER METHODS
// ============================================================================

void ModularNeuralNetwork::update_performance_metrics() {
    // Update internal performance tracking
    // This could include:
    // - Module synchronization metrics
    // - Communication efficiency
    // - Resource utilization
    
    if (modules_.size() > 0) {
        validate_modules();
    }
}

void ModularNeuralNetwork::validate_modules() {
    // Validate module states and connections
    for (auto& module : modules_) {
        if (module) {
            // Basic validation - check if module is responsive
            try {
                module->is_active();  // Simple call to test responsiveness
            } catch (const std::exception& e) {
                std::cerr << "ModularNeuralNetwork: Module validation failed for " 
                          << module->get_name() << ": " << e.what() << std::endl;
            }
        }
    }
}