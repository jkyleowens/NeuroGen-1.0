#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <vector>
#include <string>
#include <NeuroGen/Neuron.h>

class NetworkLayer {
public:
    NetworkLayer(const std::string& id);

    void addNeuron(const Neuron& neuron);
    const std::vector<Neuron>& getNeurons() const;
    const std::string& getId() const;

private:
    std::string id_;
    std::vector<Neuron> neurons_;
};

#endif // NETWORK_LAYER_H