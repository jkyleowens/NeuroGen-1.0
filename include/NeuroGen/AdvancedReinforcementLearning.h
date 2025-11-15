#ifndef ADVANCED_REINFORCEMENT_LEARNING_H
#define ADVANCED_REINFORCEMENT_LEARNING_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <NeuroGen/NeuralConstants.h>

// Forward declarations (avoiding CUDA dependencies)
struct GPUNeuronState;
struct ActorCriticState;
struct CuriosityState;
struct ValueFunction;
struct DopamineNeuron;

/**
 * @brief Advanced Reinforcement Learning with Biologically-Inspired Mechanisms
 * 
 * This system implements sophisticated reinforcement learning algorithms that
 * mirror the computational principles found in biological reward systems:
 * - Dopaminergic prediction error computation
 * - Actor-critic architecture with eligibility traces
 * - Curiosity-driven intrinsic motivation
 * - Multi-timescale learning (seconds to hours)
 * - Experience replay and consolidation
 * - Meta-learning and adaptation
 */
class AdvancedReinforcementLearning {
private:
    // GPU memory management (void* to avoid CUDA dependencies)
    void* d_da_neurons_;
    void* d_actor_critic_states_;
    void* d_curiosity_states_;
    void* d_value_functions_;
    void* d_reward_signals_;
    void* d_experience_buffer_;
    
    // Network parameters
    int num_da_neurons_;
    int num_network_neurons_;
    int num_value_functions_;
    bool cuda_initialized_;
    
    // Learning parameters
    float actor_learning_rate_;
    float critic_learning_rate_;
    float curiosity_strength_;
    float exploration_noise_;
    float experience_replay_rate_;
    
    // Performance tracking
    mutable std::mutex rl_mutex_;
    float total_reward_;
    float prediction_accuracy_;
    int learning_updates_;
    float exploration_efficiency_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    AdvancedReinforcementLearning();
    ~AdvancedReinforcementLearning();
    
    /**
     * @brief Initialize reinforcement learning system
     * @param num_da_neurons Number of dopamine neurons
     * @param num_network_neurons Total network neurons for modulation
     * @param num_value_functions Number of parallel value functions
     * @return Success status
     */
    bool initialize(int num_da_neurons, int num_network_neurons, int num_value_functions = 1);
    
    /**
     * @brief Configure comprehensive learning parameters
     * @param actor_lr Actor network learning rate
     * @param critic_lr Critic network learning rate
     * @param curiosity_strength Intrinsic motivation strength
     * @param exploration_noise Exploration noise magnitude
     */
    void configure_learning_parameters(float actor_lr, float critic_lr, 
                                     float curiosity_strength, float exploration_noise);
    
    // ========================================================================
    // MAIN LEARNING MECHANISMS - C++ WRAPPER INTERFACE
    // ========================================================================
    
    /**
     * @brief Update dopaminergic system with reward processing
     * @param reward_signal Current environmental reward
     * @param predicted_reward Value function prediction
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_dopaminergic_system(float reward_signal, float predicted_reward, 
                                   float current_time, float dt);
    
    /**
     * @brief Update actor-critic learning with experience
     * @param state_features Current state representation
     * @param action_taken Action that was executed
     * @param reward_received Reward from environment
     * @param next_state_features Subsequent state representation
     * @param dt Time step
     */
    void update_actor_critic(const std::vector<float>& state_features, int action_taken,
                           float reward_received, const std::vector<float>& next_state_features, 
                           float dt);
    
    /**
     * @brief Update curiosity-driven exploration system
     * @param environmental_features Current environmental observations
     * @param prediction_error Prediction error magnitude
     * @param dt Time step
     */
    void update_curiosity_system(const std::vector<float>& environmental_features,
                                float prediction_error, float dt);
    
    /**
     * @brief Execute experience replay for consolidation
     * @param num_replay_samples Number of experiences to replay
     * @param dt Time step
     */
    void execute_experience_replay(int num_replay_samples, float dt);
    
    /**
     * @brief Update value function approximation
     * @param state_features Current state features
     * @param target_value Target value for update
     * @param dt Time step
     */
    void update_value_function(const std::vector<float>& state_features, 
                              float target_value, float dt);
    
    // ========================================================================
    // DECISION MAKING INTERFACE
    // ========================================================================
    
    /**
     * @brief Get action probabilities from actor network
     * @param state_features Current state representation
     * @return Probability distribution over actions
     */
    std::vector<float> get_action_probabilities(const std::vector<float>& state_features) const;
    
    /**
     * @brief Get state value estimate from critic network
     * @param state_features Current state representation
     * @return Estimated state value
     */
    float get_state_value_estimate(const std::vector<float>& state_features) const;
    
    /**
     * @brief Get intrinsic motivation signal
     * @param state_features Current state
     * @return Curiosity-driven reward bonus
     */
    float get_intrinsic_motivation(const std::vector<float>& state_features) const;
    
    /**
     * @brief Sample action from policy
     * @param state_features Current state
     * @param exploration_factor Exploration intensity [0,1]
     * @return Selected action index
     */
    int sample_action(const std::vector<float>& state_features, float exploration_factor = 1.0f) const;
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get current dopamine levels across DA neurons
     * @return Vector of dopamine concentrations
     */
    std::vector<float> get_dopamine_levels() const;
    
    /**
     * @brief Get comprehensive learning performance metrics
     * @param metrics Output vector containing performance data
     */
    void get_performance_metrics(std::vector<float>& metrics) const;
    
    /**
     * @brief Get total accumulated reward
     * @return Cumulative reward received
     */
    float get_total_reward() const;
    
    /**
     * @brief Get prediction accuracy of value functions
     * @return Current prediction accuracy [0,1]
     */
    float get_prediction_accuracy() const;
    
    /**
     * @brief Get exploration efficiency metric
     * @return Exploration efficiency measure
     */
    float get_exploration_efficiency() const;
    
    /**
     * @brief Generate detailed learning report
     * @param filename Output filename for detailed analysis
     */
    void generate_learning_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save complete RL state to file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool save_rl_state(const std::string& filename) const;
    
    /**
     * @brief Load complete RL state from file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool load_rl_state(const std::string& filename);
    
    /**
     * @brief Reset all learning state to baseline
     */
    void reset_learning_state();

private:
    // ========================================================================
    // INTERNAL CUDA WRAPPER FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources for RL
     * @return Success status
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Cleanup all CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Launch dopamine update kernel (internal wrapper)
     * @param reward_signal Reward signal
     * @param predicted_reward Predicted reward
     * @param current_time Current time
     * @param dt Time step
     */
    void launch_dopamine_update_kernel(float reward_signal, float predicted_reward,
                                     float current_time, float dt);
    
    /**
     * @brief Launch actor-critic update kernel (internal wrapper)
     * @param state_features State features
     * @param action_taken Action taken
     * @param reward_received Reward received
     * @param dt Time step
     */
    void launch_actor_critic_kernel(const std::vector<float>& state_features, 
                                   int action_taken, float reward_received, float dt);
    
    /**
     * @brief Launch curiosity update kernel (internal wrapper)
     * @param environmental_features Environmental features
     * @param prediction_error Prediction error
     * @param dt Time step
     */
    void launch_curiosity_kernel(const std::vector<float>& environmental_features,
                                float prediction_error, float dt);
    
    /**
     * @brief Copy results from GPU to CPU
     */
    void copy_results_from_gpu();
    
    /**
     * @brief Validate CUDA operations
     * @param operation_name Operation name for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
    
    /**
     * @brief Store experience in replay buffer
     * @param state Current state
     * @param action Action taken
     * @param reward Reward received
     * @param next_state Next state
     */
    void store_experience(const std::vector<float>& state, int action, 
                         float reward, const std::vector<float>& next_state);
};

#endif // ADVANCED_REINFORCEMENT_LEARNING_H