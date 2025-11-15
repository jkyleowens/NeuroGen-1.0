#include "NeuroGen/TopologyGenerator.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// --- FIX: Constructor signature and member order now match the header. ---
TopologyGenerator::TopologyGenerator(const NetworkConfig& config)
    : config_(config),
      random_engine_(std::random_device{}()),
      grid_bin_size_(50.0f) // Example bin size
{}

// --- FIX: Function signatures and types now match the header. ---
void TopologyGenerator::generate_neurons(std::vector<GPUNeuronState>& neurons, std::vector<NeuronPosition>& positions, int count) {
    neurons.resize(count);
    positions.resize(count);
    std::uniform_real_distribution<float> x_dist(0, config_.network_width);
    std::uniform_real_distribution<float> y_dist(0, config_.network_height);
    std::uniform_real_distribution<float> z_dist(0, config_.network_depth);

    for (int i = 0; i < count; ++i) {
        positions[i] = {x_dist(random_engine_), y_dist(random_engine_), z_dist(random_engine_)};
        // Initialize neuron state
        neurons[i] = {}; // Zero-initialize
        neurons[i].last_spike_time = -1e9f;
        neurons[i].excitability = 1.0f;
        neurons[i].synaptic_scaling_factor = 1.0f;
    }
}

void TopologyGenerator::define_area(const std::string& name, const NeuronArea& area) {
    defined_areas_[name] = area;
}

void TopologyGenerator::build_spatial_grid(const std::vector<NeuronPosition>& positions) {
    // Implementation for building the spatial grid...
}

void TopologyGenerator::generate_synapses(
    std::vector<GPUSynapse>& synapses,
    const std::vector<NeuronPosition>& positions,
    const std::vector<ConnectionRule>& rules)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (const auto& rule : rules) {
        // --- FIX: Correctly get neuron indices for the defined areas.
        auto source_indices = get_neurons_in_area(defined_areas_[rule.source_area_name], positions);
        auto target_indices = get_neurons_in_area(defined_areas_[rule.target_area_name], positions);

        for (int source_idx : source_indices) {
            for (int target_idx : target_indices) {
                if (source_idx == target_idx) continue;
                if (dist(random_engine_) < rule.probability) {
                    GPUSynapse new_synapse;
                    initialize_synapse(new_synapse, source_idx, target_idx, rule);
                    synapses.push_back(new_synapse);
                }
            }
        }
    }
}

void TopologyGenerator::initialize_synapse(GPUSynapse& synapse, int pre_idx, int post_idx, const ConnectionRule& rule) {
    // --- FIX: Use correct members from ConnectionRule.
    std::normal_distribution<float> weight_dist(rule.weight_mean, rule.weight_std_dev);
    std::uniform_real_distribution<float> delay_dist(rule.delay_min, rule.delay_max);

    // --- Core Connectivity ---
    synapse.pre_neuron_idx = pre_idx;
    synapse.post_neuron_idx = post_idx;
    synapse.active = 1;
    synapse.post_compartment = rule.receptor_type;

    // --- Weight & Delay ---
    synapse.weight = weight_dist(random_engine_);
    synapse.delay = delay_dist(random_engine_);

    // --- FIX: Correctly initialize the restored members. ---
    synapse.eligibility_trace = 0.0f;
    synapse.plasticity_modulation = 1.0f;
    synapse.effective_weight = synapse.weight;
    synapse.last_pre_spike_time = -1e9f;
    synapse.last_post_spike_time = -1e9f;
    synapse.last_active_time = 0.0f;
    synapse.activity_metric = 0.0f;
    synapse.min_weight = 0.0f;
    synapse.max_weight = synapse.weight * 3.0f; // Example: allow 3x growth
}

std::vector<int> TopologyGenerator::get_neurons_in_area(const NeuronArea& area, const std::vector<NeuronPosition>& positions) {
   std::vector<int> indices;
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& pos = positions[i];
        if (pos.x >= area.x_min && pos.x <= area.x_max &&
            pos.y >= area.y_min && pos.y <= area.y_max &&
            pos.z >= area.z_min && pos.z <= area.z_max) {
            indices.push_back(i);
        }
    }
   return indices;
}

std::vector<int> TopologyGenerator::find_neighbors(int neuron_idx, const std::vector<NeuronPosition>& positions, float radius) {
    // A proper implementation would use the spatial grid.
    // This is a placeholder brute-force implementation.
    std::vector<int> neighbors;
    const auto& source_pos = positions[neuron_idx];
    for (size_t i = 0; i < positions.size(); ++i) {
        if (static_cast<int>(i) == neuron_idx) continue;
        const auto& target_pos = positions[i];
        float dx = source_pos.x - target_pos.x;
        float dy = source_pos.y - target_pos.y;
        float dz = source_pos.z - target_pos.z;
        if ((dx*dx + dy*dy + dz*dz) < (radius * radius)) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}