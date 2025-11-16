// Python bindings using pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

// Include NeuroGen headers
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>

namespace py = pybind11;

// Wrapper class to adapt Network interface for Python training script
class NetworkWrapper {
private:
    Network network;
    float last_reward = 0.0f;
    std::vector<float> last_output;

public:
    explicit NetworkWrapper(const NetworkConfig& config) : network(config) {}

    // Step method: takes dt and input vector, returns predicted token ID
    int step(float dt, const std::vector<float>& input_vector) {
        // Update network with input
        network.update(dt, input_vector, last_reward);

        // Get output and return first value as predicted token ID
        last_output = network.get_output();
        if (last_output.empty()) {
            return 0;
        }

        // Return rounded output as token ID (clamp to valid range)
        int predicted_id = static_cast<int>(std::round(last_output[0]));
        return std::max(0, predicted_id);
    }

    // Apply reward for learning
    void apply_reward(float reward) {
        last_reward = reward;
    }

    // Save model to file
    void save_model(const std::string& file_path) {
        if (!network.saveToFile(file_path)) {
            throw std::runtime_error("Failed to save model to " + file_path);
        }
    }

    // Load model from file
    void load_model(const std::string& file_path) {
        if (!network.loadFromFile(file_path)) {
            throw std::runtime_error("Failed to load model from " + file_path);
        }
    }

    // Get network statistics
    std::string get_stats() const {
        NetworkStats stats = network.get_stats();

        // Format stats as a simple string
        return "NetworkStats{neurons=" + std::to_string(stats.total_neurons) +
               ", synapses=" + std::to_string(stats.total_synapses) +
               ", mean_firing_rate=" + std::to_string(stats.mean_firing_rate) + "}";
    }

    // Reset network state
    void reset() {
        network.reset();
        last_reward = 0.0f;
    }
};

// PYBIND11_MODULE defines the module name and its contents
// The first parameter MUST match the filename (neural_network for neural_network.so)
PYBIND11_MODULE(neural_network, m) {
    m.doc() = "NeuroGen Neural Network Python bindings";

    // Bind NetworkConfig class
    py::class_<NetworkConfig>(m, "NetworkConfig")
        .def(py::init<>(), "Create a default NetworkConfig")

        // Core simulation parameters
        .def_readwrite("dt", &NetworkConfig::dt, "Integration time step (ms)")
        .def_readwrite("axonal_speed", &NetworkConfig::axonal_speed, "Axonal conduction speed (m/s)")

        // Spatial organization
        .def_readwrite("network_width", &NetworkConfig::network_width, "Network width (μm)")
        .def_readwrite("network_height", &NetworkConfig::network_height, "Network height (μm)")
        .def_readwrite("network_depth", &NetworkConfig::network_depth, "Network depth (μm)")

        // Neuron parameters
        .def_readwrite("num_neurons", &NetworkConfig::num_neurons, "Number of neurons")
        .def_readwrite("max_neurons", &NetworkConfig::max_neurons, "Maximum number of neurons")

        // Synapse parameters
        .def_readwrite("num_synapses", &NetworkConfig::totalSynapses, "Total number of synapses")

        // Plasticity parameters
        .def_readwrite("enable_stdp", &NetworkConfig::enable_stdp, "Enable STDP")
        .def_readwrite("stdp_learning_rate", &NetworkConfig::stdp_learning_rate, "STDP learning rate")
        .def_readwrite("learning_rate", &NetworkConfig::reward_learning_rate, "Reward learning rate")

        // Network topology
        .def_readwrite("input_size", &NetworkConfig::input_size, "Input layer size")
        .def_readwrite("hidden_size", &NetworkConfig::hidden_size, "Hidden layer size")
        .def_readwrite("output_size", &NetworkConfig::output_size, "Output layer size")

        // STDP parameters
        .def_readwrite("A_plus", &NetworkConfig::A_plus, "STDP potentiation amplitude")
        .def_readwrite("A_minus", &NetworkConfig::A_minus, "STDP depression amplitude")
        .def_readwrite("tau_plus", &NetworkConfig::tau_plus, "STDP potentiation time constant (ms)")
        .def_readwrite("tau_minus", &NetworkConfig::tau_minus, "STDP depression time constant (ms)")
        .def_readwrite("min_weight", &NetworkConfig::min_weight, "Minimum synaptic weight")
        .def_readwrite("max_weight", &NetworkConfig::max_weight, "Maximum synaptic weight")

        // Homeostatic parameters
        .def_readwrite("homeostatic_strength", &NetworkConfig::homeostatic_strength, "Homeostatic scaling strength")

        // Structural plasticity
        .def_readwrite("enable_structural_plasticity", &NetworkConfig::enable_structural_plasticity, "Enable structural plasticity")
        .def_readwrite("enable_neurogenesis", &NetworkConfig::enable_neurogenesis, "Enable neurogenesis")
        .def_readwrite("enable_pruning", &NetworkConfig::enable_pruning, "Enable pruning")

        // Topology parameters
        .def_readwrite("numColumns", &NetworkConfig::numColumns, "Number of cortical columns")
        .def_readwrite("neuronsPerColumn", &NetworkConfig::neuronsPerColumn, "Neurons per column")
        .def_readwrite("localFanOut", &NetworkConfig::localFanOut, "Local fan-out connections")
        .def_readwrite("localFanIn", &NetworkConfig::localFanIn, "Local fan-in connections")

        // Weight ranges
        .def_readwrite("wExcMin", &NetworkConfig::wExcMin, "Minimum excitatory weight")
        .def_readwrite("wExcMax", &NetworkConfig::wExcMax, "Maximum excitatory weight")
        .def_readwrite("wInhMin", &NetworkConfig::wInhMin, "Minimum inhibitory weight")
        .def_readwrite("wInhMax", &NetworkConfig::wInhMax, "Maximum inhibitory weight")

        // Delay ranges
        .def_readwrite("dMin", &NetworkConfig::dMin, "Minimum synaptic delay (ms)")
        .def_readwrite("dMax", &NetworkConfig::dMax, "Maximum synaptic delay (ms)")

        // Methods
        .def("print", &NetworkConfig::print, "Print configuration")
        .def("validate", &NetworkConfig::validate, "Validate configuration")
        .def("finalizeConfig", &NetworkConfig::finalizeConfig, "Finalize configuration")
        .def("toString", &NetworkConfig::toString, "Convert to string representation");

    // Bind NetworkStats class
    py::class_<NetworkStats>(m, "NetworkStats")
        .def(py::init<>())
        .def_readwrite("total_neurons", &NetworkStats::total_neurons)
        .def_readwrite("total_synapses", &NetworkStats::total_synapses)
        .def_readwrite("mean_firing_rate", &NetworkStats::mean_firing_rate)
        .def_readwrite("total_spike_count", &NetworkStats::total_spike_count)
        .def_readwrite("current_spike_count", &NetworkStats::current_spike_count)
        .def_readwrite("active_neuron_count", &NetworkStats::active_neuron_count)
        .def_readwrite("neuron_activity_ratio", &NetworkStats::neuron_activity_ratio)
        .def_readwrite("mean_synaptic_weight", &NetworkStats::mean_synaptic_weight)
        .def_readwrite("plasticity_rate", &NetworkStats::plasticity_rate)
        .def_readwrite("current_reward", &NetworkStats::current_reward)
        .def_readwrite("cumulative_reward", &NetworkStats::cumulative_reward)
        .def("__repr__", [](const NetworkStats& s) {
            return "NetworkStats(neurons=" + std::to_string(s.total_neurons) +
                   ", synapses=" + std::to_string(s.total_synapses) +
                   ", mean_firing_rate=" + std::to_string(s.mean_firing_rate) +
                   ", total_spikes=" + std::to_string(s.total_spike_count) + ")";
        });

    // Bind NetworkWrapper as "Network" for Python
    py::class_<NetworkWrapper>(m, "Network")
        .def(py::init<const NetworkConfig&>(),
             "Create a neural network with the given configuration",
             py::arg("config"))
        .def("step", &NetworkWrapper::step,
             "Perform one simulation step and return predicted token ID",
             py::arg("dt"), py::arg("input_vector"))
        .def("apply_reward", &NetworkWrapper::apply_reward,
             "Apply reward signal for learning",
             py::arg("reward"))
        .def("save_model", &NetworkWrapper::save_model,
             "Save network model to file",
             py::arg("file_path"))
        .def("load_model", &NetworkWrapper::load_model,
             "Load network model from file",
             py::arg("file_path"))
        .def("get_stats", &NetworkWrapper::get_stats,
             "Get network statistics as string")
        .def("reset", &NetworkWrapper::reset,
             "Reset network state");

    // Version information
    m.attr("__version__") = "1.0.0";
}
