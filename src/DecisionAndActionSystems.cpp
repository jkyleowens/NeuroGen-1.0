// ============================================================================
// DECISION MAKING AND ACTION EXECUTION SYSTEMS
// File: src/DecisionAndActionSystems.cpp
// ============================================================================

#include <NeuroGen/SpecializedModule.h>
#include <NeuroGen/AutonomousLearningAgent.h>
#include <NeuroGen/SafetyManager.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

// ============================================================================
// REMAINING SPECIALIZED MODULE PROCESSING METHODS
// ============================================================================

std::vector<float> SpecializedModule::process_motor_cortex(const std::vector<float>& motor_input) {
    // Motor cortex: Precise motor planning and execution with muscle synergies
    size_t input_size = std::min(motor_input.size(), internal_state_.size());
    
    // Update motor planning state
    for (size_t i = 0; i < input_size; ++i) {
        // Motor cortex integrates commands over time for smooth execution
        internal_state_[i] = internal_state_[i] * 0.85f + motor_input[i] * attention_weight_ * 0.15f;
    }
    
    // Generate motor commands with biological constraints
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float motor_command = 0.0f;
        
        // Integrate from corresponding internal state region
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            motor_command += internal_state_[j];
        }
        
        motor_command /= (state_end - state_start);
        
        // Apply motor activation function (sigmoidal for muscle-like response)
        output_buffer_[i] = std::tanh(motor_command * 2.0f - activation_threshold_);
        
        // Add motor noise for biological realism
        output_buffer_[i] += (rand() / float(RAND_MAX) - 0.5f) * 0.02f;
        
        // Bound motor outputs
        output_buffer_[i] = std::max(-1.0f, std::min(output_buffer_[i], 1.0f));
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_attention_system(const std::vector<float>& attention_input) {
    // Attention system: Dynamic resource allocation and focus control
    size_t input_size = std::min(attention_input.size(), internal_state_.size());
    
    // Rapid attention updates (high learning rate)
    for (size_t i = 0; i < input_size; ++i) {
        internal_state_[i] = internal_state_[i] * 0.8f + attention_input[i] * 0.2f;
    }
    
    // Compute attention weights with winner-take-all dynamics
    std::vector<float> raw_attention(output_buffer_.size());
    
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float attention_strength = 0.0f;
        
        // Compute attention from internal state
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            attention_strength += internal_state_[j] * internal_state_[j]; // Quadratic nonlinearity
        }
        
        raw_attention[i] = attention_strength / (state_end - state_start);
    }
    
    // Apply softmax for competitive attention
    float max_attention = *std::max_element(raw_attention.begin(), raw_attention.end());
    float sum_exp = 0.0f;
    
    for (float& att : raw_attention) {
        att = std::exp((att - max_attention) * 3.0f); // Temperature = 1/3
        sum_exp += att;
    }
    
    // Normalize and store in output buffer
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        output_buffer_[i] = raw_attention[i] / (sum_exp + 1e-8f);
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_reward_system(const std::vector<float>& reward_input) {
    // Reward system: Dopaminergic prediction error and value learning
    size_t input_size = std::min(reward_input.size(), internal_state_.size() / 2);
    
    // Update reward prediction state
    for (size_t i = 0; i < input_size; ++i) {
        internal_state_[i] = internal_state_[i] * 0.95f + reward_input[i] * 0.05f;
    }
    
    // Compute reward prediction and prediction error
    float predicted_reward = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        predicted_reward += internal_state_[i];
    }
    predicted_reward /= input_size;
    
    // Current reward (simplified - would come from environment)
    float current_reward = (input_size > 0) ? reward_input[0] : 0.0f;
    float prediction_error = current_reward - predicted_reward;
    
    // Update value function using TD learning
    float td_learning_rate = learning_rate_ * 5.0f; // Fast reward learning
    for (size_t i = 0; i < input_size; ++i) {
        internal_state_[i] += td_learning_rate * prediction_error * internal_state_[i];
    }
    
    // Generate dopamine-like signals
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        if (i == 0) {
            // Primary dopamine signal (prediction error)
            output_buffer_[i] = std::tanh(prediction_error * 2.0f);
        } else if (i == 1) {
            // Predicted value signal
            output_buffer_[i] = std::tanh(predicted_reward);
        } else {
            // Modulated signals for different brain regions
            float modulation = prediction_error * (1.0f + 0.1f * std::sin(i * 0.5f));
            output_buffer_[i] = std::tanh(modulation);
        }
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_working_memory(const std::vector<float>& memory_input) {
    // Working memory: Temporary information maintenance and manipulation
    size_t input_size = std::min(memory_input.size(), internal_state_.size());
    
    // Fast working memory updates with gating
    for (size_t i = 0; i < input_size; ++i) {
        // Gating function - only update if input is strong enough
        float gate_value = std::abs(memory_input[i]) > 0.3f ? 1.0f : 0.1f;
        internal_state_[i] = internal_state_[i] * (1.0f - gate_value * 0.3f) + 
                            memory_input[i] * gate_value * 0.3f;
    }
    
    // Working memory decay (forgetting)
    for (float& state : internal_state_) {
        state *= 0.98f; // Gradual decay
    }
    
    // Output maintained information
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float maintained_info = 0.0f;
        
        // Average over corresponding internal state
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            maintained_info += internal_state_[j];
        }
        
        output_buffer_[i] = maintained_info / (state_end - state_start);
    }
    
    return output_buffer_;
}

// ============================================================================
// AUTONOMOUS LEARNING AGENT - COGNITIVE PROCESSING METHODS
// ============================================================================

void AutonomousLearningAgent::process_visual_input() {
    // This method is disabled for NLP focus - no visual processing required
    return;
}

void AutonomousLearningAgent::update_working_memory() {
    if (!modules_.count("working_memory")) return;
    
    // Combine current sensory input with existing working memory
    std::vector<float> working_memory_input;
    working_memory_input.reserve(512);
    
    // Add visual context
    for (size_t i = 0; i < std::min(environmental_context_.size() / 2, size_t(256)); ++i) {
        working_memory_input.push_back(environmental_context_[i]);
    }
    
    // Add current goals
    for (size_t i = 0; i < std::min(current_goals_.size(), size_t(128)); ++i) {
        working_memory_input.push_back(current_goals_[i]);
    }
    
    // Add previous working memory content
    auto prev_working_memory = memory_system_->get_working_memory();
    for (size_t i = 0; i < std::min(prev_working_memory.size(), size_t(128)); ++i) {
        working_memory_input.push_back(prev_working_memory[i]);
    }
    
    // Process through working memory module
    float wm_attention = attention_controller_->get_attention_weight("working_memory");
    
    // Apply attention to input before processing
    std::vector<float> attended_wm_input = working_memory_input;
    for (size_t i = 0; i < attended_wm_input.size(); ++i) {
        attended_wm_input[i] *= wm_attention;
    }
    
    auto wm_output = modules_["working_memory"]->process(attended_wm_input);
    
    // Update memory system
    memory_system_->update_working_memory(wm_output);
}

void AutonomousLearningAgent::update_attention_weights() {
    // Prepare context for attention computation
    std::vector<float> attention_context;
    attention_context.reserve(256);
    
    // NLP-focused: No visual saliency in NLP mode
    
    // Add goal relevance
    for (size_t i = 0; i < std::min(current_goals_.size(), size_t(64)); ++i) {
        attention_context.push_back(current_goals_[i]);
    }
    
    // Add environmental complexity
    float env_complexity = 0.0f;
    for (float val : environmental_context_) {
        env_complexity += val * val;
    }
    attention_context.push_back(std::tanh(env_complexity / 100.0f));
    
    // Add task urgency
    attention_context.push_back(exploration_rate_); // Use exploration rate as urgency proxy
    
    // Update attention controller
    attention_controller_->update_context(attention_context);
    
    // Note: Learning system removed to avoid CUDA dependencies
    // TODO: Re-implement attention learning without CUDA when needed
}

void AutonomousLearningAgent::coordinate_modules() {
    // Get current attention weights
    auto attention_weights = attention_controller_->get_all_attention_weights();
    
    // Process each module with its attention weight and inter-module signals
    for (auto& [module_name, module] : modules_) {
        float attention_weight = attention_controller_->get_attention_weight(module_name);
        
        // Collect input from connected modules
        std::vector<float> module_input = collect_inter_module_signals(module_name);
        
        // Apply attention weighting to input
        for (size_t i = 0; i < module_input.size(); ++i) {
            module_input[i] *= attention_weight;
        }
        
        // Process the module
        auto module_output = module->process(module_input);
        
        // Send output to connected modules
        distribute_module_output(module_name, module_output);
    }
}

std::vector<float> AutonomousLearningAgent::collect_inter_module_signals(const std::string& target_module) {
    std::vector<float> combined_input;
    combined_input.reserve(512);
    
    // Collect signals from modules connected to the target
    for (const auto& [source_module, module] : modules_) {
        if (source_module != target_module) {
            // Use general get_output method instead of specific get_output_for_module
            auto signal = module->get_output();
            if (!signal.empty()) {
                combined_input.insert(combined_input.end(), signal.begin(), signal.end());
            }
        }
    }
    
    // Add environmental context relevant to the module
    if (target_module == "prefrontal_cortex") {
        // Add working memory and goals
        auto working_memory = memory_system_->get_working_memory();
        for (size_t i = 0; i < std::min(working_memory.size(), size_t(128)); ++i) {
            combined_input.push_back(working_memory[i]);
        }
        for (size_t i = 0; i < std::min(current_goals_.size(), size_t(64)); ++i) {
            combined_input.push_back(current_goals_[i]);
        }
    }
    
    return combined_input;
}

void AutonomousLearningAgent::distribute_module_output(const std::string& source_module, 
                                                      const std::vector<float>& output) {
    // Send output to all connected modules
    for (auto& [target_module, module] : modules_) {
        if (target_module != source_module) {
            module->receive_signal(output, source_module, "default");
        }
    }
    
    // Update global state based on key modules
    if (source_module == "prefrontal_cortex") {
        // Executive decisions affect global state
        size_t update_size = std::min(output.size(), global_state_.size() / 4);
        for (size_t i = 0; i < update_size; ++i) {
            global_state_[i] = global_state_[i] * 0.9f + output[i] * 0.1f;
        }
    } else if (source_module == "reward_system") {
        // Reward signals modulate global learning
        if (!output.empty()) {
            global_reward_signal_ = global_reward_signal_ * 0.95f + output[0] * 0.05f;
        }
    }
}

void AutonomousLearningAgent::make_decision() {
    // Decision-making pipeline combining multiple cognitive systems
    
    // Step 1: Get neural output from the motor cortex
    if (!modules_.count("motor_cortex")) return;
    auto neural_output = modules_["motor_cortex"]->get_output();

    // Step 2: Translate neural output into a set of action candidates
    auto action_candidates = translate_neural_output_to_actions(neural_output);

    // Step 3: Retrieve relevant memories to help evaluate candidates
    std::vector<float> current_state_summary;
    current_state_summary.reserve(256);
    for (size_t i = 0; i < std::min(global_state_.size(), size_t(128)); ++i) {
        current_state_summary.push_back(global_state_[i]);
    }
    for (size_t i = 0; i < std::min(environmental_context_.size(), size_t(128)); ++i) {
        current_state_summary.push_back(environmental_context_[i]);
    }
    auto similar_episodes = memory_system_->retrieve_similar_episodes(current_state_summary, 5);
    
    // Step 4: Evaluate actions using learned value function and memory
    auto action_values = evaluate_action_candidates(action_candidates, similar_episodes);
    
    // Step 5: Select action with exploration
    select_action_with_exploration(action_candidates, action_values);
    
    static int decision_count = 0;
    if (++decision_count % 100 == 0) {
        std::cout << "Decision System: Made decision #" << decision_count 
                  << " - Action type: " << static_cast<int>(selected_action_.type) 
                  << ", Confidence: " << selected_action_.confidence << std::endl;
    }
}

std::vector<BrowsingAction> AutonomousLearningAgent::translate_neural_output_to_actions(const std::vector<float>& neural_output) {
    std::vector<BrowsingAction> candidates;
    if (neural_output.size() < 10) return candidates; // Need enough data to decode

    // Action Type Decoding (first 5 values for 5 actions)
    ActionType decoded_type = static_cast<ActionType>(std::distance(neural_output.begin(), std::max_element(neural_output.begin(), neural_output.begin() + 5)));

    // Parameter Decoding (example using fixed positions)
    float confidence = (neural_output.size() > 5) ? (neural_output[5] + 1.0f) / 2.0f : 0.5f;
    int x_coord = (neural_output.size() > 6) ? static_cast<int>(neural_output[6] * 1920) : 0;
    int y_coord = (neural_output.size() > 7) ? static_cast<int>(neural_output[7] * 1080) : 0;
    int scroll_val = (neural_output.size() > 8) ? static_cast<int>(neural_output[8] * 500) : 100;

    BrowsingAction action;
    action.type = decoded_type;
    action.confidence = std::max(0.0f, std::min(1.0f, confidence));

    switch (decoded_type) {
        case ActionType::CLICK:
            action.x_coordinate = x_coord;
            action.y_coordinate = y_coord;
            break;
        case ActionType::SCROLL:
            action.scroll_amount = scroll_val;
            action.scroll_direction = (neural_output[8] > 0) ? ScrollDirection::DOWN : ScrollDirection::UP;
            break;
        case ActionType::TYPE:
            action.text_content = "decoded text"; // Placeholder
            break;
        case ActionType::ENTER:
        case ActionType::BACKSPACE:
            break;
    }
    candidates.push_back(action);

    // Add a few more variations for exploration
    if (exploration_rate_ > 0.3f) {
        BrowsingAction explore_action;
        explore_action.type = ActionType::CLICK;
        explore_action.x_coordinate = rand() % 1920;
        explore_action.y_coordinate = rand() % 1080;
        explore_action.confidence = exploration_rate_ * 0.5f;
        candidates.push_back(explore_action);
    }

    return candidates;
}

std::vector<float> AutonomousLearningAgent::evaluate_action_candidates(
    const std::vector<BrowsingAction>& candidates,
    const std::vector<MemorySystem::MemoryTrace>& similar_episodes) {
    
    std::vector<float> values(candidates.size());
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& action = candidates[i];
        float value = 0.0f;
        
        // Base value from action type
        switch (action.type) {
            case ActionType::CLICK:
                value = 0.7f;
                break;
            case ActionType::SCROLL:
                value = 0.4f;
                break;
            case ActionType::TYPE:
                value = 0.8f;
                break;
            case ActionType::ENTER:
                value = 0.6f;
                break;
            case ActionType::BACKSPACE:
                value = 0.5f;
                break;
        }
        
        // Adjust based on confidence
        value *= (0.5f + 0.5f * action.confidence);
        
        // Adjust based on similar episodes
        for (const auto& episode : similar_episodes) {
            if (!episode.action_vector.empty() && episode.action_vector.size() > 2) {
                // Simple action similarity check
                float action_similarity = 0.0f;
                if (static_cast<int>(action.type) < episode.action_vector.size()) {
                    action_similarity = episode.action_vector[static_cast<int>(action.type)];
                }
                
                // Weight by episode outcome
                value += action_similarity * episode.reward * 0.1f;
            }
        }
        
        // Add exploration bonus
        value += exploration_rate_ * 0.1f;
        
        values[i] = value;
    }
    
    return values;
}

void AutonomousLearningAgent::select_action_with_exploration(
    const std::vector<BrowsingAction>& candidates, const std::vector<float>& values) {

    if (candidates.empty()) {
        // No action to select, just return. The agent will reconsider in the next cycle.
        return;
    }
    
    // Softmax action selection with temperature
    float temperature = exploration_rate_ * 2.0f + 0.1f; 
    std::vector<float> probabilities(values.size());
    
    float max_value = *std::max_element(values.begin(), values.end());
    
    float sum_exp = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        probabilities[i] = std::exp((values[i] - max_value) / temperature);
        sum_exp += probabilities[i];
    }
    
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float random_value = dist(gen);
    float cumulative_prob = 0.0f;
    
    for (size_t i = 0; i < candidates.size(); i++) {
        cumulative_prob += probabilities[i];
        if (random_value <= cumulative_prob) {
            // Directly return the chosen candidate to preserve all its data
            selected_action_ = candidates[i];
            return;
        }
    }
    
    // Fallback to last action
    selected_action_ = candidates.back();
}

// NOTE: execute_action() is now defined in AutonomousLearningAgent.cpp to avoid multiple definitions

void AutonomousLearningAgent::execute_click_action() {
    std::cout << "Executing CLICK at (" << selected_action_.x_coordinate << ", " << selected_action_.y_coordinate
              << ") with confidence " << selected_action_.confidence << std::endl;

    if (!SafetyManager::getInstance().isActionSafe(selected_action_)) {
        std::cout << "Action blocked by safety manager" << std::endl;
        return;
    }

    bool success = false;
    if (input_controller_) {
        success = input_controller_->clickMouse(selected_action_.x_coordinate, selected_action_.y_coordinate);
    }
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.1f;
    }
}

void AutonomousLearningAgent::execute_scroll_action() {
    std::cout << "Executing SCROLL at (" << selected_action_.x_coordinate << ", " << selected_action_.y_coordinate 
              << ") with confidence " << selected_action_.confidence << std::endl;
    
    if (!SafetyManager::getInstance().isActionSafe(selected_action_)) {
        std::cout << "Action blocked by safety manager" << std::endl;
        return;
    }

    bool success = false;
    if (input_controller_) {
        success = input_controller_->scrollMouse(selected_action_.x_coordinate,
                                                 selected_action_.y_coordinate,
                                                 selected_action_.scroll_amount);
    }
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.05f;
    }
}

void AutonomousLearningAgent::execute_type_action() {
    std::cout << "Executing TYPE: '" << selected_action_.text_content
              << "' with confidence " << selected_action_.confidence << std::endl;

    bool success = false;
    if (input_controller_) {
        success = input_controller_->typeText(selected_action_.text_content);
    }
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.15f;
    }
}

void AutonomousLearningAgent::execute_enter_action() {
    std::cout << "Executing ENTER with confidence " << selected_action_.confidence << std::endl;

    bool success = false;
    if (input_controller_) {
        success = input_controller_->typeText("\n");
    }
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.1f;
    }
}

void AutonomousLearningAgent::execute_backspace_action() {
    std::cout << "Executing BACKSPACE with confidence " << selected_action_.confidence << std::endl;

    bool success = false;
    if (input_controller_) {
        success = input_controller_->typeText("\b");
    }
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.05f;
    }
}

std::vector<float> AutonomousLearningAgent::convert_action_to_motor_command(const BrowsingAction& action) {
    std::vector<float> motor_command(64, 0.0f); // Motor cortex expects 64-dimensional input
    
    // Encode action type
    if (static_cast<size_t>(action.type) < motor_command.size()) {
        motor_command[static_cast<size_t>(action.type)] = 1.0f;
    }
    
    // Encode spatial coordinates (normalized)
    if (motor_command.size() > 10) {
        motor_command[5] = action.x_coordinate / 1920.0f; // Normalize x coordinate
        motor_command[6] = action.y_coordinate / 1080.0f; // Normalize y coordinate
        motor_command[7] = action.confidence;   // Action confidence
    }
    
    // Add motor noise for biological realism
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.02f);
    
    for (size_t i = 8; i < motor_command.size(); ++i) {
        motor_command[i] = noise(gen);
    }
    
    return motor_command;
}

void AutonomousLearningAgent::learn_from_feedback() {
    // Comprehensive learning from action outcomes
    
    // Prepare experience for storage
    std::vector<float> current_state = global_state_;
    std::vector<float> action_vector(5, 0.0f);
    action_vector[static_cast<int>(selected_action_.type)] = 1.0f;
    
    // Compute reward signal based on action outcome
    float action_reward = compute_action_reward();
    
    // Store episode in memory
    memory_system_->store_episode(current_state, action_vector, action_reward, 
                                 selected_action_.confidence);
    
    // Note: Learning system updates removed to avoid CUDA dependencies
    // TODO: Re-implement learning updates without CUDA when needed
    
    // Update performance metrics
    metrics_.average_reward = metrics_.average_reward * 0.99f + action_reward * 0.01f;
    
    // Adapt exploration rate based on recent performance
    adapt_exploration_rate();
    
    // Apply learning to specific modules
    apply_modular_learning(action_reward);
}

float AutonomousLearningAgent::compute_action_reward() {
    float reward = 0.0f;
    
    // Base reward from action success
    bool action_successful = (metrics_.total_actions > 0) ? 
        (static_cast<float>(metrics_.successful_actions) / metrics_.total_actions > 0.5f) : false;
    
    if (action_successful) {
        reward += 0.1f;
    } else {
        reward -= 0.05f;
    }
    
    // Reward based on action type effectiveness
    switch (selected_action_.type) {
        case ActionType::CLICK:
            reward += selected_action_.confidence * 0.1f;
            break;
        case ActionType::TYPE:
            reward += selected_action_.confidence * 0.15f;
            break;
        case ActionType::SCROLL:
            reward += selected_action_.confidence * 0.05f;
            break;
        case ActionType::ENTER:
            reward += selected_action_.confidence * 0.08f;
            break;
        case ActionType::BACKSPACE:
            reward += selected_action_.confidence * 0.03f;
            break;
    }
    
    // Penalty for excessive exploration
    if (exploration_rate_ > 0.5f) {
        reward -= 0.02f;
    }
    
    // Note: Learning progress reward removed due to learning_system_ removal
    // TODO: Re-implement learning progress tracking without CUDA dependencies
    
    return std::max(-0.5f, std::min(reward, 0.5f)); // Bound reward
}

void AutonomousLearningAgent::adapt_exploration_rate() {
    float success_rate = (metrics_.total_actions > 0) ? 
        static_cast<float>(metrics_.successful_actions) / metrics_.total_actions : 0.5f;
    
    // Decrease exploration if doing well, increase if struggling
    if (success_rate > 0.7f) {
        exploration_rate_ *= 0.995f; // Slowly decrease exploration
    } else if (success_rate < 0.3f) {
        exploration_rate_ *= 1.005f; // Slowly increase exploration
    }
    
    // Bound exploration rate
    exploration_rate_ = std::max(0.05f, std::min(exploration_rate_, 0.8f));
}

void AutonomousLearningAgent::apply_modular_learning(float reward) {
    // Note: learning_system_ was removed to avoid CUDA dependencies
    // TODO: Re-implement modular learning without CUDA when needed
    
    // Apply learning to each module based on its contribution to the action
    std::vector<std::string> relevant_modules;
    
    switch (selected_action_.type) {
        case ActionType::CLICK:
        case ActionType::SCROLL:
            relevant_modules = {"motor_cortex", "attention_system"};
            break;
        case ActionType::TYPE:
        case ActionType::ENTER:
        case ActionType::BACKSPACE:
            relevant_modules = {"prefrontal_cortex", "working_memory", "motor_cortex"};
            break;
    }
    
    // Note: Module-specific learning would be applied here if learning_system_ was available
}

void AutonomousLearningAgent::update_global_state() {
    // Integrate information from all cognitive systems into global state
    
    // Decay existing state
    for (float& state : global_state_) {
        state *= 0.98f;
    }
    
    // Add environmental context
    for (size_t i = 0; i < std::min(environmental_context_.size(), global_state_.size() / 4); ++i) {
        global_state_[i] += environmental_context_[i] * 0.02f;
    }
    
    // Add working memory content
    auto working_memory = memory_system_->get_working_memory();
    size_t wm_offset = global_state_.size() / 4;
    for (size_t i = 0; i < std::min(working_memory.size(), global_state_.size() / 4); ++i) {
        if (wm_offset + i < global_state_.size()) {
            global_state_[wm_offset + i] += working_memory[i] * 0.03f;
        }
    }
    
    // Add attention state
    auto attention_weights = attention_controller_->get_all_attention_weights();
    size_t att_offset = global_state_.size() / 2;
    for (size_t i = 0; i < std::min(attention_weights.size(), global_state_.size() / 8); ++i) {
        if (att_offset + i < global_state_.size()) {
            global_state_[att_offset + i] = attention_weights[i];
        }
    }
    
    // Add reward history
    size_t reward_offset = 3 * global_state_.size() / 4;
    if (reward_offset < global_state_.size()) {
        global_state_[reward_offset] = global_reward_signal_;
    }
    
    // Bound global state values
    for (float& state : global_state_) {
        state = std::max(-2.0f, std::min(state, 2.0f));
    }
}

void AutonomousLearningAgent::consolidate_learning() {
    // Periodic learning consolidation and memory management
    
    std::cout << "Learning consolidation: Integrating recent experiences..." << std::endl;
    
    // Consolidate memories
    memory_system_->consolidateMemories();
    
    // Transfer learning between modules
    transfer_knowledge_between_modules();
    
    // Note: Learning system parameter updates removed to avoid CUDA dependencies
    // TODO: Re-implement learning parameter adjustment without CUDA when needed
    
    std::cout << "Learning consolidation complete." << std::endl;
}

void AutonomousLearningAgent::transfer_knowledge_between_modules() {
    // Simple knowledge transfer between related modules
    
    // Hippocampus-Prefrontal transfer (memory-guided decisions)
    if (modules_.count("hippocampus") && modules_.count("prefrontal_cortex")) {
        auto memory_state = modules_["hippocampus"]->get_output();
        std::vector<float> memory_patterns(memory_state.begin(),
                                         memory_state.begin() + std::min(memory_state.size(), size_t(256)));
        modules_["prefrontal_cortex"]->receive_signal(memory_patterns, "hippocampus", "default");
    }
}