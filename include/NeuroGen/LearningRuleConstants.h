#ifndef LEARNING_RULE_CONSTANTS_H
#define LEARNING_RULE_CONSTANTS_H

#include <string>

// >>> FIX: Replaced conflicting C-style macros with a modern C++ struct.
// This prevents macro-related name clashes and provides a clean, type-safe
// way to configure learning rules on the CPU side.

/**
 * @struct STDPConfig
 * @brief Holds parameters for a Spike-Timing-Dependent Plasticity learning rule.
 */
struct STDPConfig {
    float A_plus = 0.01f;     // Potentiation learning rate
    float A_minus = 0.012f;   // Depression learning rate
    float tau_plus = 20.0f;   // Time constant for pre-before-post events (ms)
    float tau_minus = 20.0f;  // Time constant for post-before-pre events (ms)
};

/**
 * @struct OjaRuleConfig
 * @brief Holds parameters for Oja's Rule, a form of Hebbian learning.
 */
struct OjaRuleConfig {
    float learning_rate = 0.001f; // Learning rate (gamma)
};

/**
 * @struct BCMConfig
 * @brief Holds parameters for the BCM (Bienenstock-Cooper-Munro) learning rule.
 */
struct BCMConfig {
    float learning_rate = 0.01f;  // Learning rate (eta)
    float theta_m_initial = 10.0f; // Initial modification threshold
    float tau_theta = 1000.0f;     // Time constant for the sliding threshold (ms)
};

#endif // LEARNING_RULE_CONSTANTS_H