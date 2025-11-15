#include <NeuroGen/Network.h>
#include <NeuroGen/NeuralModule.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR  
// ============================================================================

Network::Network(const NetworkConfig& config)
    : config_(config), module_(nullptr), random_engine_(std::random_device{}()) {
    
    std::cout << "🧠 Initializing breakthrough neural network..." << std::endl;
    std::cout << "   • Target neurons: " << config.hidden_size << std::endl;
    std::cout << "   • Structural plasticity: " << (config.enable_structural_plasticity ? "ENABLED" : "disabled") << std::endl;
    
    initialize_neurons();
    initialize_synapses();
    
    // Initialize network statistics
    stats_.active_neuron_count = neurons_.size();
    stats_.active_synapses = synapses_.size();
    stats_.total_synapses = synapses_.size();
    stats_.simulation_steps = 0;
    stats_.mean_firing_rate = 0.0f;
    stats_.network_entropy = 0.0f;
    
    std::cout << "✅ Network initialized with " << neurons_.size() << " neurons and " 
              << synapses_.size() << " synapses" << std::endl;
}

Network::~Network() = default;

// ============================================================================
// CORE SIMULATION INTERFACE
// ============================================================================

void Network::update(float dt, const std::vector<float>& input_currents, float reward) {
    // Update internal simulation state
    stats_.simulation_steps++;
    
    // Core neural dynamics
    update_neurons(dt, input_currents);
    update_synapses(dt, reward);
    
    // Advanced biological mechanisms
    if (config_.enable_structural_plasticity) {
        structural_plasticity();
    }
    
    // Update network statistics
    update_stats(dt);
}

std::vector<float> Network::get_output() const {
    std::vector<float> outputs;
    
    if (neurons_.empty()) {
        return outputs;
    }
    
    // Extract output from the last neurons in the network (output layer)
    size_t output_start = std::max(0, static_cast<int>(neurons_.size()) - config_.output_size);
    outputs.reserve(config_.output_size);
    
    for (size_t i = output_start; i < neurons_.size(); i++) {
        if (neurons_[i]) {
            // Use firing rate as output
            float firing_rate = calculateNeuronFiringRate(*neurons_[i]);
            outputs.push_back(firing_rate);
        } else {
            outputs.push_back(0.0f);
        }
    }
    
    return outputs;
}

void Network::reset() {
    std::cout << "🔄 Resetting neural network state..." << std::endl;
    
    // Reset all neurons
    for (auto& neuron : neurons_) {
        if (neuron) {
            // Reset neuron to default state
            neuron.reset(); // Assuming Neuron has a reset method
        }
    }
    
    // Reset all synapses
    for (auto& synapse : synapses_) {
        if (synapse) {
            synapse->eligibility_trace = 0.0;
            synapse->activity_metric = 0.0;
            synapse->last_pre_spike = -1000.0;
            synapse->last_post_spike = -1000.0;
        }
    }
    
    // Reset statistics
    stats_.simulation_steps = 0;
    stats_.mean_firing_rate = 0.0f;
    stats_.network_entropy = 0.0f;
    
    std::cout << "✅ Network reset completed" << std::endl;
}

// ============================================================================
// NETWORK CONSTRUCTION INTERFACE
// ============================================================================

void Network::add_neuron(std::unique_ptr<Neuron> neuron) {
    if (neuron) {
        size_t neuron_id = neuron->get_id();
        neuron_map_[neuron_id] = neuron.get();
        neurons_.push_back(std::move(neuron));
    }
}

void Network::add_synapse(std::unique_ptr<Synapse> synapse) {
    if (synapse) {
        size_t synapse_id = synapse->id;
        synapse_map_[synapse_id] = synapse.get();
        
        // Update connection maps
        outgoing_synapse_map_[synapse->pre_neuron_id].push_back(synapse.get());
        incoming_synapse_map_[synapse->post_neuron_id].push_back(synapse.get());
        
        synapses_.push_back(std::move(synapse));
    }
}

Synapse* Network::createSynapse(size_t source_neuron_id, size_t target_neuron_id, 
                               const std::string& type, int delay, float weight) {
    // Validate neuron IDs
    if (neuron_map_.find(source_neuron_id) == neuron_map_.end() ||
        neuron_map_.find(target_neuron_id) == neuron_map_.end()) {
        std::cerr << "Error: Invalid neuron IDs for synapse creation: " 
                  << source_neuron_id << " -> " << target_neuron_id << std::endl;
        return nullptr;
    }
    
    // Generate unique synapse ID
    size_t synapse_id = synapses_.size();
    
    // Determine receptor type and compartment based on synapse type
    size_t receptor_index = 0; // Default to excitatory
    std::string compartment = "soma"; // Default compartment
    
    if (type == "inhibitory") {
        receptor_index = 1;
        weight = -std::abs(weight); // Ensure inhibitory weights are negative
    } else {
        weight = std::abs(weight); // Ensure excitatory weights are positive
    }
    
    // Create synapse with biological parameters
    auto synapse = std::make_unique<Synapse>(
        synapse_id, source_neuron_id, target_neuron_id, compartment, 
        receptor_index, weight, static_cast<double>(delay)
    );
    
    Synapse* synapse_ptr = synapse.get();
    add_synapse(std::move(synapse));
    
    return synapse_ptr;
}

// ============================================================================
// NETWORK ACCESS INTERFACE
// ============================================================================


std::vector<Synapse*> Network::getOutgoingSynapses(size_t neuron_id) const {
    auto it = outgoing_synapse_map_.find(neuron_id);
    return (it != outgoing_synapse_map_.end()) ? it->second : std::vector<Synapse*>();
}

std::vector<Synapse*> Network::getIncomingSynapses(size_t neuron_id) const {
    auto it = incoming_synapse_map_.find(neuron_id);
    return (it != incoming_synapse_map_.end()) ? it->second : std::vector<Synapse*>();
}

void Network::set_module(NeuralModule* module) {
    module_ = module;
    std::cout << "🔗 Neural module association established" << std::endl;
}

NetworkStats Network::get_stats() const {
    return stats_;
}

// ============================================================================
// PERSISTENCE METHODS
// ============================================================================

bool Network::saveToFile(const std::string& file_path) const {
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Cannot open file for writing: " << file_path << std::endl;
        return false;
    }

    // Write neuron data
    size_t num_neurons = neurons_.size();
    ofs.write(reinterpret_cast<const char*>(&num_neurons), sizeof(num_neurons));
    for (const auto& neuron : neurons_) {
        size_t neuron_id = neuron->get_id();
        ofs.write(reinterpret_cast<const char*>(&neuron_id), sizeof(neuron_id));
        // Note: NeuronParams are not saved, assuming they are constant
        float potential = neuron->get_potential();
        ofs.write(reinterpret_cast<const char*>(&potential), sizeof(potential));
    }

    // Write synapse data
    size_t num_synapses = synapses_.size();
    ofs.write(reinterpret_cast<const char*>(&num_synapses), sizeof(num_synapses));
    for (const auto& synapse : synapses_) {
        ofs.write(reinterpret_cast<const char*>(&synapse->id), sizeof(synapse->id));
        ofs.write(reinterpret_cast<const char*>(&synapse->pre_neuron_id), sizeof(synapse->pre_neuron_id));
        ofs.write(reinterpret_cast<const char*>(&synapse->post_neuron_id), sizeof(synapse->post_neuron_id));
        ofs.write(reinterpret_cast<const char*>(&synapse->weight), sizeof(synapse->weight));
        ofs.write(reinterpret_cast<const char*>(&synapse->axonal_delay), sizeof(synapse->axonal_delay));
    }

    return true;
}

bool Network::loadFromFile(const std::string& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Cannot open file for reading: " << file_path << std::endl;
        return false;
    }

    // Clear existing network
    neurons_.clear();
    synapses_.clear();
    neuron_map_.clear();
    incoming_synapse_map_.clear();
    outgoing_synapse_map_.clear();

    // Read neuron data
    size_t num_neurons;
    ifs.read(reinterpret_cast<char*>(&num_neurons), sizeof(num_neurons));
    NeuronParams params; // Assuming default params
    for (size_t i = 0; i < num_neurons; ++i) {
        size_t neuron_id;
        float potential;
        ifs.read(reinterpret_cast<char*>(&neuron_id), sizeof(neuron_id));
        ifs.read(reinterpret_cast<char*>(&potential), sizeof(potential));
        auto neuron = std::make_unique<Neuron>(neuron_id, params);
        neuron->set_potential(potential); // Need to add set_potential to Neuron class
        add_neuron(std::move(neuron));
    }

    // Read synapse data
    size_t num_synapses;
    ifs.read(reinterpret_cast<char*>(&num_synapses), sizeof(num_synapses));
    for (size_t i = 0; i < num_synapses; ++i) {
        size_t id, pre_id, post_id;
        double weight, delay;
        ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
        ifs.read(reinterpret_cast<char*>(&pre_id), sizeof(pre_id));
        ifs.read(reinterpret_cast<char*>(&post_id), sizeof(post_id));
        ifs.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        ifs.read(reinterpret_cast<char*>(&delay), sizeof(delay));
        createSynapse(pre_id, post_id, "excitatory", static_cast<int>(delay), static_cast<float>(weight));
    }

    rebuild_connection_maps();
    return true;
}

// ============================================================================
// PRIVATE IMPLEMENTATION: INITIALIZATION
// ============================================================================

void Network::initialize_neurons() {
    std::cout << "🧬 Initializing " << config_.hidden_size << " neurons with biological diversity..." << std::endl;
    
    // Create neurobiologically diverse neuron population
    NeuronParams params;
    std::uniform_real_distribution<float> variability(-0.05f, 0.05f);
    
    for (size_t i = 0; i < config_.hidden_size; ++i) {
        // Add biological variability to parameters
        NeuronParams varied_params = params;
        varied_params.a += variability(random_engine_);
        varied_params.b += variability(random_engine_);
        varied_params.c += variability(random_engine_);
        varied_params.d += variability(random_engine_);
        
        add_neuron(std::make_unique<Neuron>(i, varied_params));
    }
    
    std::cout << "✅ Neurons initialized with biological parameter diversity" << std::endl;
}

void Network::initialize_synapses() {
    std::cout << "🔗 Initializing synaptic connections..." << std::endl;
    
    if (neurons_.empty()) {
        std::cout << "⚠️  No neurons available for synapse creation" << std::endl;
        return;
    }
    
    // Create initial connectivity with enhanced parameters for version 0.5.5
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(config_.min_weight, config_.max_weight);
    std::uniform_int_distribution<int> delay_dist(1, 5);
    
    // Enhanced target synapses for version 0.5.5 - aim for realistic connectivity
    size_t target_synapses = std::min(static_cast<size_t>(config_.totalSynapses), 
                                     neurons_.size() * neurons_.size() / 4); // More aggressive targeting
    
    // Ensure minimum connectivity for small networks
    size_t min_synapses = neurons_.size() * 3; // At least 3 connections per neuron on average
    target_synapses = std::max(target_synapses, min_synapses);
    
    std::cout << "🎯 Target synapses: " << target_synapses << " (for " << neurons_.size() << " neurons)" << std::endl;
    
    size_t created_synapses = 0;
    for (size_t pre = 0; pre < neurons_.size() && created_synapses < target_synapses; pre++) {
        for (size_t post = 0; post < neurons_.size() && created_synapses < target_synapses; post++) {
            if (pre == post) continue; // No self-connections
            
            // Enhanced connection probability for biological realism with better connectivity
            float connection_prob = calculateConnectionProbability(pre, post);
            
            if (prob_dist(random_engine_) < connection_prob) {
                std::string synapse_type = (prob_dist(random_engine_) < config_.exc_ratio) ? 
                                          "excitatory" : "inhibitory";
                float weight = weight_dist(random_engine_);
                int delay = delay_dist(random_engine_);
                
                if (createSynapse(pre, post, synapse_type, delay, weight)) {
                    created_synapses++;
                }
            }
        }
    }
    
    std::cout << "✅ Created " << created_synapses << " synaptic connections" << std::endl;
    
    // If we still have very few connections, create some guaranteed ones
    if (created_synapses < min_synapses / 2) {
        std::cout << "🔧 Creating additional connections to ensure network functionality..." << std::endl;
        
        size_t additional_created = 0;
        std::uniform_int_distribution<size_t> neuron_dist(0, neurons_.size() - 1);
        
        while (additional_created < min_synapses / 2 && (created_synapses + additional_created) < target_synapses) {
            size_t pre = neuron_dist(random_engine_);
            size_t post = neuron_dist(random_engine_);
            
            if (pre != post && !synapseExists(pre, post)) {
                std::string synapse_type = (prob_dist(random_engine_) < config_.exc_ratio) ? 
                                          "excitatory" : "inhibitory";
                float weight = weight_dist(random_engine_);
                int delay = delay_dist(random_engine_);
                
                if (createSynapse(pre, post, synapse_type, delay, weight)) {
                    additional_created++;
                }
            }
        }
        
        std::cout << "✅ Created " << additional_created << " additional synaptic connections" << std::endl;
    }
}

// ============================================================================
// PRIVATE IMPLEMENTATION: NEURAL DYNAMICS
// ============================================================================

void Network::update_neurons(float dt, const std::vector<float>& input_currents) {
    if (neurons_.empty()) return;
    
    // Apply external inputs to input neurons
    size_t input_neurons = std::min(input_currents.size(), neurons_.size());
    
    for (size_t i = 0; i < input_neurons; i++) {
        if (neurons_[i] && i < input_currents.size()) {
            // Apply input current (implementation depends on Neuron interface)
            // neurons_[i]->add_input_current(input_currents[i]);
        }
    }
    
    // Update all neurons
    for (auto& neuron : neurons_) {
        if (neuron) {
            // Calculate total synaptic input
            float total_input = calculateTotalSynapticInput(neuron->get_id());
            neuron->update(dt, total_input);
        }
    }
}

void Network::update_synapses(float dt, float reward) {
    if (synapses_.empty()) return;
    
    // Update synaptic dynamics and plasticity
    for (auto& synapse : synapses_) {
        if (synapse) {
            updateSynapticPlasticity(*synapse, dt, reward);
        }
    }
}

void Network::structural_plasticity() {
    if (!config_.enable_structural_plasticity) return;
    
    // Implement synaptic pruning and growth
    prune_synapses();
    grow_synapses();
}

void Network::update_stats(float dt) {
    // Update network activity statistics
    stats_.simulation_steps++;
    
    // Calculate mean firing rate
    float total_firing_rate = 0.0f;
    int active_neurons = 0;
    
    for (const auto& neuron : neurons_) {
        if (neuron) {
            float firing_rate = calculateNeuronFiringRate(*neuron);
            total_firing_rate += firing_rate;
            if (firing_rate > 0.1f) active_neurons++;
        }
    }
    
    stats_.mean_firing_rate = (neurons_.empty()) ? 0.0f : total_firing_rate / neurons_.size();
    stats_.active_neuron_count = active_neurons;
    stats_.neuron_activity_ratio = (neurons_.empty()) ? 0.0f : 
                                  static_cast<float>(active_neurons) / neurons_.size();
    
    // Calculate network entropy
    if (stats_.neuron_activity_ratio > 0.0f && stats_.neuron_activity_ratio < 1.0f) {
        float p = stats_.neuron_activity_ratio;
        stats_.network_entropy = -(p * std::log2(p) + (1.0f - p) * std::log2(1.0f - p));
    }
    
    // Update synapse statistics
    int active_synapses = 0;
    for (const auto& synapse : synapses_) {
        if (synapse && synapse->activity_metric > 0.01) {
            active_synapses++;
        }
    }
    stats_.active_synapses = active_synapses;
}

// ============================================================================
// PRIVATE IMPLEMENTATION: STRUCTURAL PLASTICITY
// ============================================================================

void Network::prune_synapses() {
    // Remove weak or inactive synapses
    auto it = std::remove_if(synapses_.begin(), synapses_.end(),
        [this](const std::unique_ptr<Synapse>& synapse) {
            return synapse && shouldPruneSynapse(*synapse);
        });
    
    size_t pruned_count = std::distance(it, synapses_.end());
    if (pruned_count > 0) {
        synapses_.erase(it, synapses_.end());
        std::cout << "🌿 Pruned " << pruned_count << " weak synapses" << std::endl;
        
        // Rebuild connection maps
        rebuild_connection_maps();
    }
}

void Network::grow_synapses() {
    // Enhanced synaptic growth for version 0.5.5
    if (synapses_.size() >= static_cast<size_t>(config_.totalSynapses)) return;
    
    size_t max_new_synapses = std::max(static_cast<size_t>(5), synapses_.size() / 50); // Grow 2% per step, minimum 5
    size_t created = 0;
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(config_.min_weight * 0.5f, config_.max_weight * 0.7f);
    std::uniform_int_distribution<size_t> neuron_dist(0, neurons_.size() - 1);
    
    for (size_t attempt = 0; attempt < max_new_synapses * 20 && created < max_new_synapses; attempt++) {
        size_t pre_id = neuron_dist(random_engine_);
        size_t post_id = neuron_dist(random_engine_);
        
        if (pre_id != post_id && !synapseExists(pre_id, post_id)) {
            // Enhanced growth: prefer active neurons but also allow spontaneous growth
            bool should_connect = false;
            
            if (isNeuronActive(pre_id) && isNeuronActive(post_id)) {
                // High probability for active-active connections
                should_connect = prob_dist(random_engine_) < 0.8f;
            } else if (isNeuronActive(pre_id) || isNeuronActive(post_id)) {
                // Medium probability for partially active connections
                should_connect = prob_dist(random_engine_) < 0.4f;
            } else {
                // Low probability for inactive connections (exploration)
                should_connect = prob_dist(random_engine_) < 0.1f;
            }
            
            if (should_connect) {
                float weight = weight_dist(random_engine_);
                std::string type = (prob_dist(random_engine_) < config_.exc_ratio) ? 
                                  "excitatory" : "inhibitory";
                
                if (createSynapse(pre_id, post_id, type, 1, weight)) {
                    created++;
                }
            }
        }
    }
    
    if (created > 0) {
        std::cout << "🌱 Grew " << created << " new synapses (total: " << synapses_.size() << ")" << std::endl;
    }
}

bool Network::shouldPruneSynapse(const Synapse& synapse) const {
    // Multi-criteria pruning decision
    bool is_weak = std::abs(synapse.weight) < config_.min_weight * 0.1;
    bool is_inactive = synapse.activity_metric < 0.001;
    bool is_old_and_unused = (synapse.activity_metric < 0.01) && 
                            (stats_.simulation_steps - synapse.formation_time > 10000);
    
    return is_weak && (is_inactive || is_old_and_unused);
}

// ============================================================================
// PRIVATE IMPLEMENTATION: UTILITY FUNCTIONS
// ============================================================================

float Network::calculateConnectionProbability(size_t pre_id, size_t post_id) const {
    // Enhanced connection probability for version 0.5.5
    constexpr float BASE_PROB = 0.15f; // 15% base connection probability
    constexpr float DISTANCE_SCALE = 20.0f; // Characteristic distance scale
    
    float distance = std::abs(static_cast<float>(post_id) - static_cast<float>(pre_id));
    float distance_factor = std::exp(-distance / DISTANCE_SCALE);
    
    // Add small-world topology bias for enhanced connectivity
    float random_factor = 0.05f; // 5% random long-range connections
    
    // Boost probability for version 0.5.5 features
    float enhanced_prob = (BASE_PROB * distance_factor + random_factor) * 1.5f;
    
    return std::min(0.4f, enhanced_prob); // Cap at 40% for biological realism
}

float Network::calculateTotalSynapticInput(size_t neuron_id) const {
    float total_input = 0.0f;
    
    auto incoming = getIncomingSynapses(neuron_id);
    for (const auto& synapse : incoming) {
        if (synapse) {
            // Check if presynaptic neuron recently spiked
            auto pre_neuron = get_neuron(synapse->pre_neuron_id);
            if (pre_neuron && pre_neuron->has_spiked()) {
                total_input += synapse->weight;
            }
        }
    }
    
    return total_input;
}

// In src/Network.cpp

float Network::calculateNeuronFiringRate(const Neuron& neuron) const {
    // Biologically-realistic firing rate calculation
    constexpr float BASELINE_RATE = 2.0f;  // Hz - cortical baseline
    constexpr float MAX_RATE = 100.0f;     // Hz - physiological maximum
    constexpr float THRESHOLD_VOLTAGE = -55.0f; // mV - typical spike threshold
    
    if (neuron.has_spiked()) {
        // Active neuron: rate depends on membrane potential dynamics
        float potential_factor = std::tanh((neuron.get_potential() - THRESHOLD_VOLTAGE) / 20.0f);
        return BASELINE_RATE + (MAX_RATE - BASELINE_RATE) * std::max(0.0f, potential_factor);
    }
    
    // Subthreshold activity contributes to background rate
    float subthreshold_factor = std::max(0.0f, (neuron.get_potential() + 70.0f) / 50.0f);
    return BASELINE_RATE * subthreshold_factor;
}

void Network::updateSynapticPlasticity(Synapse& synapse, float dt, float reward) {
    // Update activity metric
    auto pre_neuron = get_neuron(synapse.pre_neuron_id);
    auto post_neuron = get_neuron(synapse.post_neuron_id);
    
    if (pre_neuron && post_neuron) {
        bool pre_spiked = pre_neuron->has_spiked();
        bool post_spiked = post_neuron->has_spiked();
        
        // Update activity
        if (pre_spiked || post_spiked) {
            synapse.activity_metric = synapse.activity_metric * 0.999f + 0.001f;
        } else {
            synapse.activity_metric *= 0.9999f;
        }
        
        // Simple STDP rule
        if (pre_spiked && post_spiked) {
            synapse.weight += 0.01f * reward; // Reward-modulated plasticity
            float min_weight = static_cast<float>(config_.min_weight);
            float max_weight = static_cast<float>(config_.max_weight);  
            float current_weight = static_cast<float>(synapse.weight);

            synapse.weight = std::max(min_weight, std::min(max_weight, current_weight));
        }
    }
}

bool Network::isNeuronActive(size_t neuron_id) const {
    auto neuron = get_neuron(neuron_id);
    return neuron && calculateNeuronFiringRate(*neuron) > 1.0f;
}

void Network::rebuild_connection_maps() {
    // Clear existing maps
    outgoing_synapse_map_.clear();
    incoming_synapse_map_.clear();
    synapse_map_.clear();
    
    // Rebuild from current synapse list
    for (auto& synapse : synapses_) {
        if (synapse) {
            synapse_map_[synapse->id] = synapse.get();
            outgoing_synapse_map_[synapse->pre_neuron_id].push_back(synapse.get());
            incoming_synapse_map_[synapse->post_neuron_id].push_back(synapse.get());
        }
    }
}