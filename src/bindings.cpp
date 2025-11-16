/*
 * ======================================================================
 * src/bindings.cpp
 * * This file creates the Python bindings for your C++ CUDA-accelerated
 * network. It uses pybind11 to expose your NetworkCUDA_Interface
 * class to Python so it can be imported as a module.
 * ======================================================================
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed to convert C++ vectors/lists to Python lists
#include <pybind11/iostream.h> // For redirecting C++ cout to Python
#include <NeuroGen/cuda/NetworkCUDA_Interface.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/NetworkConfig.h> // Ensure this is includable
#include <NeuroGen/NetworkStats.h>   // Ensure this is includable

namespace py = pybind11;

// Simple wrapper class for Python bindings
// This extends NetworkCUDA_Interface with Python-friendly methods
class PyNetworkCUDA_Interface : public NetworkCUDA_Interface {
public:
    // Use constructor from base class
    using NetworkCUDA_Interface::NetworkCUDA_Interface;

    // Placeholder methods for future implementation
    // These will be implemented when needed for training functionality
};


// This is the C++ macro that creates the Python module
// The first argument ("neurogen_backend") is the name you use in `import`
PYBIND11_MODULE(neurogen_backend, m) {
    m.doc() = "High-performance C++/CUDA backend for NeuroGen";
    
    // Allow C++ cout/cerr to be printed in Python
    py::add_ostream_redirect(m, "ostream_redirect");

    // Expose the NetworkStats struct
    py::class_<NetworkStats>(m, "NetworkStats")
        .def(py::init<>())
        .def_readwrite("total_neurons", &NetworkStats::total_neurons)
        .def_readwrite("total_synapses", &NetworkStats::total_synapses)
        .def_readwrite("mean_firing_rate", &NetworkStats::mean_firing_rate)
        .def_readwrite("current_time_ms", &NetworkStats::current_time_ms)
        .def_readwrite("simulation_steps", &NetworkStats::simulation_steps);
        
    // Expose the NetworkConfig struct
    py::class_<NetworkConfig>(m, "NetworkConfig")
        .def(py::init<>())
        .def_readwrite("dt", &NetworkConfig::dt)
        .def_readwrite("num_neurons", &NetworkConfig::num_neurons)
        .def_readwrite("input_size", &NetworkConfig::input_size)
        .def_readwrite("output_size", &NetworkConfig::output_size)
        .def_readwrite("hidden_size", &NetworkConfig::hidden_size);

    // Expose the NetworkCUDA_Interface class
    // We bind our extended class `PyNetworkCUDA_Interface`
    py::class_<PyNetworkCUDA_Interface>(m, "NetworkCUDA_Interface")
        // Constructor takes config, neurons, and synapses
        .def(py::init<const NetworkConfig&, 
                      const std::vector<GPUNeuronState>&, 
                      const std::vector<GPUSynapse>&>(),
             py::arg("config"),
             py::arg("neurons"),
             py::arg("synapses"),
             "Create a new CUDA-accelerated neural network")
        
        // Expose the step function
        .def("step", &PyNetworkCUDA_Interface::step,
             py::arg("current_time"),
             py::arg("dt"),
             py::arg("reward"),
             py::arg("inputs"),
             "Run one simulation step")
        
        // Expose the get_stats function
        .def("get_stats", &PyNetworkCUDA_Interface::get_stats, 
             "Get network statistics")
        
        // Expose the get_network_state function
        .def("get_network_state", &PyNetworkCUDA_Interface::get_network_state,
             py::arg("neurons"),
             py::arg("synapses"),
             "Get current network state");
}