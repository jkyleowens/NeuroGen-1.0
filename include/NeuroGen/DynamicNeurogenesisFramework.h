#ifndef DYNAMIC_NEUROGENESIS_FRAMEWORK_H
#define DYNAMIC_NEUROGENESIS_FRAMEWORK_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <NeuroGen/NeuralConstants.h>

// Forward declarations (no CUDA dependencies)
struct GPUNeuronState;
struct NeuralProgenitor;
struct DevelopmentalTrajectory;
struct ValueFunction;

/**
 * @brief Dynamic Neurogenesis Framework - C++ Only Interface
 * 
 * Implements biologically-realistic neurogenesis mechanisms including:
 * - Neural progenitor cell dynamics
 * - Activity-dependent neurogenesis
 * - Developmental trajectory control
 * - Spatial organization and migration
 * - Experience-dependent plasticity
 */
class DynamicNeurogenesisFramework {
private:
    // GPU memory management (void* to avoid CUDA dependencies)
    void* d_neural_progenitors_;
    void* d_neurons_;
    void* d_developmental_trajectories_;
    void* d_value_functions_;
    void* d_neurogenesis_controller_;
    
    // Network parameters
    int max_progenitors_;
    int current_progenitors_;
    int max_neurons_;
    int current_neurons_;
    bool cuda_initialized_;
    
    // Neurogenesis parameters
    float neurogenesis_rate_;
    float activity_threshold_;
    float spatial_competition_radius_;
    float experience_dependence_;
    
    // Performance tracking
    mutable std::mutex neurogenesis_mutex_;
    int neurons_generated_;
    int progenitors_created_;
    float average_maturation_time_;

public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE
    // ========================================================================
    
    DynamicNeurogenesisFramework();
    ~DynamicNeurogenesisFramework();
    
    /**
     * @brief Initialize neurogenesis framework
     * @param max_progenitors Maximum number of progenitor cells
     * @param max_neurons Maximum number of mature neurons
     * @param initial_neurogenesis_rate Initial rate of neurogenesis
     * @return Success status
     */
    bool initialize(int max_progenitors, int max_neurons, float initial_neurogenesis_rate);
    
    /**
     * @brief Configure neurogenesis parameters
     * @param neurogenesis_rate Rate of new neuron generation
     * @param activity_threshold Activity threshold for neurogenesis
     * @param spatial_radius Spatial competition radius
     * @param experience_dep Experience dependence factor
     */
    void configure_neurogenesis_parameters(float neurogenesis_rate, float activity_threshold,
                                         float spatial_radius, float experience_dep);
    
    // ========================================================================
    // MAIN NEUROGENESIS MECHANISMS - C++ WRAPPER INTERFACE
    // ========================================================================
    
    /**
     * @brief Update neurogenesis control mechanisms
     * @param current_time Current simulation time
     * @param dt Time step
     * @param global_activity_level Overall network activity
     */
    void update_neurogenesis_control(float current_time, float dt, float global_activity_level);
    
    /**
     * @brief Execute progenitor cell spawning
     * @param current_time Current simulation time
     * @param dt Time step
     * @param environmental_factors Environmental influences
     */
    void update_progenitor_spawning(float current_time, float dt, 
                                   const std::vector<float>& environmental_factors);
    
    /**
     * @brief Update progenitor cell development
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_progenitor_development(float current_time, float dt);
    
    /**
     * @brief Execute neuron differentiation from progenitors
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_neuron_differentiation(float current_time, float dt);
    
    /**
     * @brief Update developmental trajectories
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_developmental_trajectories(float current_time, float dt);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get current number of active progenitors
     * @return Number of progenitor cells
     */
    int get_current_progenitor_count() const;
    
    /**
     * @brief Get current number of mature neurons
     * @return Number of differentiated neurons
     */
    int get_current_neuron_count() const;
    
    /**
     * @brief Get neurogenesis rate statistics
     * @param stats Output vector for neurogenesis statistics
     */
    void get_neurogenesis_statistics(std::vector<float>& stats) const;
    
    /**
     * @brief Get developmental stage distribution
     * @return Vector of counts per developmental stage
     */
    std::vector<int> get_developmental_stage_distribution() const;
    
    /**
     * @brief Generate neurogenesis report
     * @param filename Output filename for detailed report
     */
    void generate_neurogenesis_report(const std::string& filename) const;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save neurogenesis state to file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool save_neurogenesis_state(const std::string& filename) const;
    
    /**
     * @brief Load neurogenesis state from file
     * @param filename Base filename for state files
     * @return Success status
     */
    bool load_neurogenesis_state(const std::string& filename);
    
    /**
     * @brief Reset neurogenesis to baseline state
     */
    void reset_neurogenesis_state();

private:
    // ========================================================================
    // INTERNAL CUDA WRAPPER FUNCTIONS (NO DEVICE CODE IN HEADER)
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources for neurogenesis
     * @return Success status
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Cleanup CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Launch neurogenesis control kernel (wrapper)
     * @param current_time Current time
     * @param dt Time step
     * @param global_activity Global activity level
     */
    void launch_neurogenesis_control_kernel(float current_time, float dt, float global_activity);
    
    /**
     * @brief Launch progenitor spawning kernel (wrapper)
     * @param current_time Current time
     * @param dt Time step
     * @param environmental_factors Environmental influences
     */
    void launch_progenitor_spawning_kernel(float current_time, float dt,
                                          const std::vector<float>& environmental_factors);
    
    /**
     * @brief Copy statistics from GPU to CPU
     */
    void copy_statistics_from_gpu();
    
    /**
     * @brief Validate CUDA operations
     * @param operation_name Operation name for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
};

#endif // DYNAMIC_NEUROGENESIS_FRAMEWORK_H
