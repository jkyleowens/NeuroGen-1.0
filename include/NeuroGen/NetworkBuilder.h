#ifndef NETWORK_BUILDER_H
#define NETWORK_BUILDER_H

#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * @class NetworkBuilder
 * @brief Provides a fluent interface for constructing complex Network objects.
 *
 * This class implements the builder pattern to allow for step-by-step
 * definition of network layers and their connections before building the
 * final Network object.
 */
class NetworkBuilder {
private:
    // Internal structs to hold definitions until build() is called
    struct LayerDefinition {
        std::string name;
        int num_neurons;
        std::string type; // e.g., "excitatory", "inhibitory"
    };

    struct ConnectionDefinition {
        std::string from_layer;
        std::string to_layer;
        float weight;
        float delay;
    };

public:
    NetworkBuilder();

    /**
     * @brief Sets the overall network configuration.
     * @param config The NetworkConfig object.
     * @return A reference to the builder for chaining.
     */
    NetworkBuilder& withConfig(const NetworkConfig& config);

    /**
     * @brief Adds a layer of neurons to the network definition.
     * @param name A unique name for the layer (e.g., "input", "hidden1").
     * @param num_neurons The number of neurons in this layer.
     * @param type A string describing the neuron type (optional).
     * @return A reference to the builder for chaining.
     */
    NetworkBuilder& addLayer(const std::string& name, int num_neurons, const std::string& type = "excitatory");

    /**
     * @brief Defines a dense (all-to-all) connection between two layers.
     * @param from_layer The name of the source layer.
     * @param to_layer The name of the target layer.
     * @param weight The initial weight for all synapses in this connection.
     * @param delay The transmission delay for all synapses in this connection (optional).
     * @return A reference to the builder for chaining.
     */
    NetworkBuilder& connectLayers(const std::string& from_layer, const std::string& to_layer, float weight, float delay = 1.0f);

    /**
     * @brief Constructs the Network object based on the provided definitions.
     * @return A unique_ptr to the newly created Network.
     */
    std::unique_ptr<Network> build();

private:
    NetworkConfig config_;
    std::vector<LayerDefinition> layer_defs_;
    std::vector<ConnectionDefinition> connection_defs_;
};

#endif // NETWORK_BUILDER_H