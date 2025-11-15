#pragma once

#include <string>
#include <cstdint>

/**
 * @class InterModuleConnection
 * @brief Represents a directed connection between two modules in the brain architecture.
 *
 * This class defines the properties of a synaptic link, including the source
 * and target modules, the connection strength (weight), and the signal
 * propagation time (delay).
 */
class InterModuleConnection {
public:
    // The unique identifier of the source module.
    std::string source_module_id;

    // The unique identifier of the target module.
    std::string target_module_id;

    // The strength or efficacy of the connection. Can be positive (excitatory) or negative (inhibitory).
    float weight;

    // The time delay in milliseconds for a signal to travel from the source to the target.
    uint32_t delay_ms;

    /**
     * @brief Default constructor.
     */
    InterModuleConnection() = default;

    /**
     * @brief Constructs an InterModuleConnection with specified properties.
     * @param source The ID of the module where the connection originates.
     * @param target The ID of the module where the connection terminates.
     * @param conn_weight The weight of the connection.
     * @param conn_delay_ms The propagation delay in milliseconds.
     */
    InterModuleConnection(std::string source, std::string target, float conn_weight, uint32_t conn_delay_ms)
        : source_module_id(std::move(source)),
          target_module_id(std::move(target)),
          weight(conn_weight),
          delay_ms(conn_delay_ms) {}
};