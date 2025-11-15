#include "NeuroGen/NetworkBuilder.h"
#include <iostream>
#include <numeric>

NetworkBuilder::NetworkBuilder() {
    // Correctly initialize the config using the new member names.
    // We set the *maximum* allowable neurons here. The actual number will be
    // determined when build() is called.
    config_.max_neurons = 4096; // A default maximum
    
    // FIX: Use the 'enable_structural_plasticity' member which is now present.
    config_.enable_structural_plasticity = false;
}

NetworkBuilder& NetworkBuilder::withConfig(const NetworkConfig& config) {
    config_ = config;
    return *this;
}

NetworkBuilder& NetworkBuilder::addLayer(const std::string& name, int num_neurons, const std::string& type) {
    layer_defs_.push_back({name, num_neurons, type});
    return *this;
}

NetworkBuilder& NetworkBuilder::connectLayers(const std::string& from_layer, const std::string& to_layer, float weight, float delay) {
    connection_defs_.push_back({from_layer, to_layer, weight, delay});
    return *this;
}

std::unique_ptr<Network> NetworkBuilder::build() {
    // 1. Calculate the total number of neurons needed from all layer definitions.
    size_t total_neurons = 0;
    for (const auto& layer_def : layer_defs_) {
        total_neurons += layer_def.num_neurons;
    }

    // Check if the required neurons exceed the configured maximum.
    if (total_neurons > config_.max_neurons) {
        std::cerr << "Error: The defined layers require " << total_neurons 
                  << " neurons, which exceeds the configured maximum of " 
                  << config_.max_neurons << "." << std::endl;
        return nullptr;
    }
    
    // The Network class itself will use this to know how many neurons to create.
    // This is a conceptual fix assuming Network's constructor uses this field.
    // If Network's constructor takes a neuron count, we'd pass total_neurons there.
    // For now, we'll assume the config needs the *actual* number.
    // Let's create a temporary config for the network constructor.
    NetworkConfig build_config = config_;
    
    // FIX: Use the 'num_neurons' member which is now present.
    build_config.num_neurons = total_neurons;


    // 2. Create the network object.
    auto network = std::make_unique<Network>(build_config);

    // 3. Partition the neuron IDs by layer for easy lookup.
    std::unordered_map<std::string, std::vector<size_t>> layer_neuron_ids;
    size_t current_neuron_id = 0;
    for (const auto& layer_def : layer_defs_) {
        std::vector<size_t> ids;
        ids.reserve(layer_def.num_neurons);
        for (int i = 0; i < layer_def.num_neurons; ++i) {
            ids.push_back(current_neuron_id++);
        }
        layer_neuron_ids[layer_def.name] = ids;
    }

    // 4. Create the synapse connections using the Network's public interface.
    for (const auto& conn_def : connection_defs_) {
        if (layer_neuron_ids.find(conn_def.from_layer) == layer_neuron_ids.end() ||
            layer_neuron_ids.find(conn_def.to_layer) == layer_neuron_ids.end()) {
            std::cerr << "Warning: Layer not found for connection from '" << conn_def.from_layer
                      << "' to '" << conn_def.to_layer << "'. Skipping." << std::endl;
            continue;
        }

        const auto& from_ids = layer_neuron_ids[conn_def.from_layer];
        const auto& to_ids = layer_neuron_ids[conn_def.to_layer];

        for (size_t from_id : from_ids) {
            for (size_t to_id : to_ids) {
                network->createSynapse(from_id, to_id, "excitatory", static_cast<int>(conn_def.delay), conn_def.weight);
            }
        }
    }

    std::cout << "Network built successfully." << std::endl;
    
    // FIX: Correctly get stats using the public member variables.
    NetworkStats final_stats = network->get_stats();
    std::cout << "Final Stats -> Neurons: " << final_stats.active_neuron_count
              << ", Synapses: " << final_stats.active_synapses << std::endl;

    return network;
}