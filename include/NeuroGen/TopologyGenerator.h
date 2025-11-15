#ifndef TOPOLOGY_GENERATOR_H
#define TOPOLOGY_GENERATOR_H

#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/cuda/GPUNeuralStructures.h"

// --- FIX: Defined all necessary helper structs here for clarity and correctness. ---

/** @brief Represents the 3D position of a neuron. */
struct NeuronPosition {
    float x, y, z;
};

/** @brief Defines a rectangular area of neurons. */
struct NeuronArea {
    float x_min, x_max, y_min, y_max, z_min, z_max;
};

/** @brief Defines a rule for connecting two groups of neurons. */
struct ConnectionRule {
    std::string source_area_name;
    std::string target_area_name;
    int receptor_type = 0; // e.g., 0 for AMPA, 1 for GABA
    float probability = 0.1f;
    float weight_mean = 0.1f;
    float weight_std_dev = 0.02f;
    float delay_min = 1.0f;
    float delay_max = 5.0f;
};
// --- END FIX ---

class TopologyGenerator {
public:
    TopologyGenerator(const NetworkConfig& config);

    void generate_neurons(std::vector<GPUNeuronState>& neurons, std::vector<NeuronPosition>& positions, int count);
    void define_area(const std::string& name, const NeuronArea& area);
    void generate_synapses(std::vector<GPUSynapse>& synapses, const std::vector<NeuronPosition>& positions, const std::vector<ConnectionRule>& rules);

private:
    void build_spatial_grid(const std::vector<NeuronPosition>& positions);
    std::vector<int> get_neurons_in_area(const NeuronArea& area, const std::vector<NeuronPosition>& positions);
    std::vector<int> find_neighbors(int neuron_idx, const std::vector<NeuronPosition>& positions, float radius);
    void initialize_synapse(GPUSynapse& synapse, int pre_idx, int post_idx, const ConnectionRule& rule);

    const NetworkConfig& config_;
    std::unordered_map<std::string, NeuronArea> defined_areas_;
    std::mt19937 random_engine_;
    // Spatial grid for accelerating neighbor searches
    std::unordered_map<int, std::vector<int>> spatial_grid_;
    float grid_bin_size_ = 50.0f;
};

#endif // TOPOLOGY_GENERATOR_H