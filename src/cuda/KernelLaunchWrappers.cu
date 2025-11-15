#include "NeuroGen/cuda/KernelLaunchWrappers.cuh"

// Include all the necessary kernel headers
#include "NeuroGen/cuda/IonChannelInitialization.cuh"
#include "NeuroGen/cuda/NeuronUpdateKernel.cuh"
#include "NeuroGen/cuda/CalciumDiffusionKernel.cuh"
#include "NeuroGen/cuda/EnhancedSTDPKernel.cuh"
#include "NeuroGen/cuda/EligibilityAndRewardKernels.cuh"
#include "NeuroGen/cuda/HomeostaticMechanismsKernel.cuh"

#include <iostream>

#define THREADS_PER_BLOCK 256

namespace KernelLaunchWrappers {

// (Other wrapper implementations remain the same)
void initialize_ion_channels(GPUNeuronState* neurons, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ionChannelInitializationKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, num_neurons);
    cudaDeviceSynchronize();
}

void update_neuron_states(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    neuronUpdateKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, current_time, dt, num_neurons);
}

void update_calcium_dynamics(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    calciumDiffusionKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, current_time, dt, num_neurons);
}

void run_stdp_and_eligibility(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses)
{
    const int num_blocks = (num_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    enhancedSTDPKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, neurons, current_time, dt, num_synapses);
}

void apply_reward_and_adaptation(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses)
{
    const int num_blocks = (num_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    applyRewardKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, reward, dt, num_synapses);
    adaptNeuromodulationKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, neurons, reward, num_synapses, current_time);
}


// --- FIX: Wrapper function now accepts and passes current_time. ---
void run_homeostatic_mechanisms(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float current_time,
    int num_neurons,
    int num_synapses)
{
    const int neuron_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // The kernel call now has the correct number of arguments.
    synapticScalingKernel<<<neuron_blocks, THREADS_PER_BLOCK>>>(neurons, synapses, num_neurons, num_synapses, current_time);
    intrinsicPlasticityKernel<<<neuron_blocks, THREADS_PER_BLOCK>>>(neurons, num_neurons);
}

} // namespace KernelLaunchWrappers