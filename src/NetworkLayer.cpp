#include <NeuroGen/NetworkLayer.h>

NetworkLayer::NetworkLayer(const std::string& id) : id_(id) {}

void NetworkLayer::addNeuron(const Neuron& neuron) {
    neurons_.push_back(neuron);
}

const std::vector<Neuron>& NetworkLayer::getNeurons() const {
    return neurons_;
}

const std::string& NetworkLayer::getId() const {
    return id_;
}