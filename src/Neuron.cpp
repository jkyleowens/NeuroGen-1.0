#include "NeuroGen/Neuron.h"
#include <algorithm> // For std::max

// --- Constructor ---
// Initializes all state variables and stores the parameter configuration.
Neuron::Neuron(size_t id, const NeuronParams& params)
    : id_(id),
      params_(params),
      potential_(params_.c), // Start at resting/reset potential
      recovery_var_(params_.b * params_.c),
      last_spike_time_(-1e9f), // Initialize to a long time ago
      has_spiked_(false)
{}

// --- Update Function ---
void Neuron::update(float dt, float total_input_current) {
    has_spiked_ = false; // Reset spike flag at the start of the update

    // --- FIX: All constants are now accessed from the params_ member struct. ---

    // 1. Check if neuron is in a refractory period
    // This requires tracking the current simulation time, which we approximate here.
    // A more robust solution would pass current_time into the update function.
    // if (current_time < last_spike_time_ + params_.absolute_refractory_period) {
    //     return;
    // }

    // 2. Update membrane potential and recovery variable (Izhikevich model)
    // dV/dt = 0.04V^2 + 5V + 140 - u + I
    // du/dt = a(bV - u)
    float v = potential_;
    float u = recovery_var_;

    potential_ += dt * (0.04f * v * v + 5.0f * v + 140.0f - u + total_input_current);
    recovery_var_ += dt * (params_.a * (params_.b * v - u));

    // 3. Check for spike firing
    if (potential_ >= params_.spike_threshold) {
        potential_ = params_.c; // Reset potential
        recovery_var_ += params_.d;
        has_spiked_ = true;
        // last_spike_time_ = current_time; // Would be set here
    }
}

// --- Getter Functions ---
bool Neuron::has_spiked() const {
    return has_spiked_;
}

size_t Neuron::get_id() const {
    return id_;
}

float Neuron::get_potential() const {
    return potential_;
}

void Neuron::set_potential(float potential) {
    potential_ = potential;
}