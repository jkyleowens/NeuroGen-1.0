#ifndef NEURAL_CONSTANTS_H
#define NEURAL_CONSTANTS_H

// ============================================================================
// NEURAL RECEPTOR AND CHANNEL CONSTANTS
// ============================================================================

// Receptor type enumeration and constants
#define NUM_RECEPTOR_TYPES 8
#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3
#define RECEPTOR_GLYCINE 4
#define RECEPTOR_ACETYLCHOLINE 5
#define RECEPTOR_DOPAMINE 6
#define RECEPTOR_SEROTONIN 7

// Neural developmental stages
#define NEURAL_STAGE_PROGENITOR 0
#define NEURAL_STAGE_DIFFERENTIATION 1
#define NEURAL_STAGE_MIGRATION 2
#define NEURAL_STAGE_MATURATION 3
#define NEURAL_STAGE_ADULT 4
#define NEURAL_STAGE_SENESCENCE 5

// Compartment types
#define COMPARTMENT_SOMA 0
#define COMPARTMENT_BASAL 1
#define COMPARTMENT_APICAL 2
#define COMPARTMENT_SPINE 3
#define COMPARTMENT_AXON 4

// Maximum structure sizes
#define MAX_COMPARTMENTS 8
#define MAX_DENDRITIC_SPIKES 4
#define MAX_SYNAPSES_PER_NEURON 1000
#define MAX_NEURAL_PROGENITORS 10000

// Biophysical constants
#define SPIKE_PEAK 30.0f
#define REVERSAL_POTENTIAL_EXCITATORY 0.0f
#define REVERSAL_POTENTIAL_INHIBITORY -70.0f

// Time constants (ms)
#define TAU_MEMBRANE 20.0f
#define TAU_CALCIUM 50.0f
#define TAU_PLASTICITY 1000.0f
#define TAU_DEVELOPMENT 86400000.0f  // 24 hours in ms

// Learning constants
#define STDP_LEARNING_RATE 0.01f
#define BCM_LEARNING_RATE 0.001f
#define HOMEOSTATIC_RATE 0.0001f
#define NEUROGENESIS_RATE 0.0001f

// Utility macros for device/host compatibility
#ifdef __CUDACC__
    #define DEVICE_HOST __device__ __host__
    #define DEVICE_ONLY __device__
    #define CUDA_MIN(a, b) fminf(a, b)
    #define CUDA_MAX(a, b) fmaxf(a, b)
#else
    #define DEVICE_HOST
    #define DEVICE_ONLY
    #define CUDA_MIN(a, b) std::min(a, b)
    #define CUDA_MAX(a, b) std::max(a, b)
#endif

#endif // NEURAL_CONSTANTS_H