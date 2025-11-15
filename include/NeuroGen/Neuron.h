#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstddef>

/**
 * @struct NeuronParams
 * @brief Holds all biophysical parameters for the CPU-side neuron model.
 */
struct NeuronParams {
    // Izhikevich model parameters
    float a = 0.02f;
    float b = 0.2f;
    float c = -65.0f; // Reset potential
    float d = 8.0f;   // Recovery variable reset adjustment

    // Firing and refractory properties
    float spike_threshold = 30.0f;
    float absolute_refractory_period = 2.0f; // ms
};

/**
 * @class Neuron
 * @brief Represents a single neuron for CPU-based simulations or initialization.
 */
class Neuron {
public:
    /**
     * @brief Constructs a Neuron.
     * @param id A unique identifier for the neuron.
     * @param params A struct containing the neuron's biophysical parameters.
     */
    Neuron(size_t id, const NeuronParams& params);

    /**
     * @brief Updates the neuron's state for a single time step.
     * @param dt The simulation time step in milliseconds.
     * @param total_input_current The sum of all synaptic currents.
     */
    void update(float dt, float total_input_current);

    /**
     * @brief Checks if the neuron has fired a spike in the last update.
     * @return True if the neuron has spiked, false otherwise.
     */
    bool has_spiked() const;

    /**
     * @brief Gets the neuron's unique identifier.
     * @return The neuron's ID.
     */
    size_t get_id() const;

    /**
     * @brief Gets the neuron's current membrane potential.
     * @return The membrane potential in mV.
     */
    float get_potential() const;

    /**
     * @brief Sets the neuron's current membrane potential.
     * @param potential The membrane potential in mV.
     */
    void set_potential(float potential);

private:
    size_t id_;
    NeuronParams params_; // Stores the neuron's configuration

    // State variables
    float potential_;       // Membrane potential (V_m)
    float recovery_var_;    // Recovery variable (u)
    float last_spike_time_;
    bool has_spiked_;
};

#endif // NEURON_H