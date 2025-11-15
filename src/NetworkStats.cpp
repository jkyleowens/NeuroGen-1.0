#include <NeuroGen/NetworkStats.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ============================================================================
// GLOBAL STATISTICS INSTANCE
// ============================================================================

#ifdef __CUDACC__
__managed__ NetworkStats g_stats;
#else
NetworkStats g_stats;
#endif

// ============================================================================
// IMPLEMENTATION OF COMPLEX METHODS
// ============================================================================

std::string NetworkStats::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    oss << "=== NeuroGen Network Statistics ===\n";
    oss << "Simulation Time: " << current_time_ms << " ms (Steps: " << simulation_steps << ")\n";
    oss << "Total Runtime: " << total_simulation_time << " ms\n\n";
    
    // Core Activity Metrics
    oss << "--- Neural Activity ---\n";
    oss << "Mean Firing Rate: " << mean_firing_rate << " Hz\n";
    oss << "Active Neurons: " << active_neuron_count << " (" << (neuron_activity_ratio * 100.0f) << "%)\n";
    oss << "Current Spikes: " << current_spike_count << " (Total: " << total_spike_count << ")\n";
    oss << "Network Synchrony: " << network_synchrony << "\n";
    oss << "Population Vector: " << population_vector_strength << "\n\n";
    
    // Synaptic Connectivity
    oss << "--- Synaptic Properties ---\n";
    oss << "Active Synapses: " << active_synapses << "/" << total_synapses;
    if (total_synapses > 0) {
        oss << " (" << (static_cast<float>(active_synapses) / total_synapses * 100.0f) << "%)";
    }
    oss << "\nMean Weight: " << mean_synaptic_weight << " ± " << synaptic_weight_std << "\n";
    oss << "E/I Ratio: " << excitation_inhibition_ratio << "\n";
    oss << "Plasticity Rate: " << plasticity_rate << "\n";
    oss << "Connection Density: " << connection_density << "\n\n";
    
    // Neuromodulation
    oss << "--- Neuromodulation ---\n";
    oss << "Dopamine: " << dopamine_level << "\n";
    oss << "Acetylcholine: " << acetylcholine_level << "\n";
    oss << "Serotonin: " << serotonin_level << "\n";
    oss << "Norepinephrine: " << norepinephrine_level << "\n";
    oss << "GABA: " << gaba_level << "\n";
    oss << "Glutamate: " << glutamate_level << "\n";
    oss << "Neuromod Balance: " << neuromod_balance << "\n\n";
    
    // Learning and Adaptation
    oss << "--- Learning System ---\n";
    oss << "Current Reward: " << current_reward << " (Avg: " << average_reward << ")\n";
    oss << "Cumulative Reward: " << cumulative_reward << "\n";
    oss << "Learning Rate: " << learning_rate << "\n";
    oss << "Network Entropy: " << network_entropy << "\n";
    oss << "Adaptation Rate: " << adaptation_rate << "\n\n";
    
    // Cortical Columns (if present)
    if (!column_activity.empty()) {
        oss << "--- Cortical Architecture ---\n";
        oss << "Columns: " << column_activity.size() << "\n";
        float mean_col_activity = 0.0f;
        for (float activity : column_activity) {
            mean_col_activity += activity;
        }
        mean_col_activity /= column_activity.size();
        oss << "Mean Column Activity: " << mean_col_activity << "\n\n";
    }
    
    // Performance and Stability
    oss << "--- Performance ---\n";
    oss << "Kernel Time: " << kernel_execution_time << " ms\n";
    oss << "Memory Transfer: " << memory_transfer_time << " ms\n";
    oss << "Simulation Speed: " << simulation_speed_factor << "x real-time\n";
    oss << "Memory Usage: " << (memory_usage_bytes / (1024.0f * 1024.0f)) << " MB\n";
    oss << "GPU Utilization: " << gpu_utilization << "%\n";
    oss << "Numerical Stability: " << numerical_stability << "\n";
    oss << "Network Health: " << (getHealthScore() > 0.7f ? "HEALTHY" : "WARNING") << "\n";
    oss << "CUDA Errors: " << cuda_error_count << "\n\n";
    
    // Efficiency Scores
    oss << "--- Efficiency Metrics ---\n";
    oss << "Performance Score: " << (getPerformanceScore() * 100.0f) << "%\n";
    oss << "Health Score: " << (getHealthScore() * 100.0f) << "%\n";
    
    return oss.str();
}

std::string NetworkStats::toJSON() const {
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    
    json << "{\n";
    json << "  \"simulation\": {\n";
    json << "    \"current_time_ms\": " << current_time_ms << ",\n";
    json << "    \"total_simulation_time\": " << total_simulation_time << ",\n";
    json << "    \"simulation_steps\": " << simulation_steps << "\n";
    json << "  },\n";
    
    json << "  \"activity\": {\n";
    json << "    \"mean_firing_rate\": " << mean_firing_rate << ",\n";
    json << "    \"firing_rate_std\": " << firing_rate_std << ",\n";
    json << "    \"current_spike_count\": " << current_spike_count << ",\n";
    json << "    \"total_spike_count\": " << total_spike_count << ",\n";
    json << "    \"active_neuron_count\": " << active_neuron_count << ",\n";
    json << "    \"neuron_activity_ratio\": " << neuron_activity_ratio << ",\n";
    json << "    \"network_synchrony\": " << network_synchrony << ",\n";
    json << "    \"population_vector_strength\": " << population_vector_strength << "\n";
    json << "  },\n";
    
    json << "  \"synaptic\": {\n";
    json << "    \"total_synapses\": " << total_synapses << ",\n";
    json << "    \"active_synapses\": " << active_synapses << ",\n";
    json << "    \"mean_synaptic_weight\": " << mean_synaptic_weight << ",\n";
    json << "    \"synaptic_weight_std\": " << synaptic_weight_std << ",\n";
    json << "    \"plasticity_rate\": " << plasticity_rate << ",\n";
    json << "    \"excitation_inhibition_ratio\": " << excitation_inhibition_ratio << ",\n";
    json << "    \"connection_density\": " << connection_density << ",\n";
    json << "    \"clustering_coefficient\": " << clustering_coefficient << ",\n";
    json << "    \"average_path_length\": " << average_path_length << "\n";
    json << "  },\n";
    
    json << "  \"neuromodulation\": {\n";
    json << "    \"dopamine_level\": " << dopamine_level << ",\n";
    json << "    \"acetylcholine_level\": " << acetylcholine_level << ",\n";
    json << "    \"serotonin_level\": " << serotonin_level << ",\n";
    json << "    \"norepinephrine_level\": " << norepinephrine_level << ",\n";
    json << "    \"gaba_level\": " << gaba_level << ",\n";
    json << "    \"glutamate_level\": " << glutamate_level << ",\n";
    json << "    \"neuromod_balance\": " << neuromod_balance << "\n";
    json << "  },\n";
    
    json << "  \"learning\": {\n";
    json << "    \"current_reward\": " << current_reward << ",\n";
    json << "    \"cumulative_reward\": " << cumulative_reward << ",\n";
    json << "    \"average_reward\": " << average_reward << ",\n";
    json << "    \"learning_rate\": " << learning_rate << ",\n";
    json << "    \"prediction_error\": " << prediction_error << ",\n";
    json << "    \"information_capacity\": " << information_capacity << ",\n";
    json << "    \"network_entropy\": " << network_entropy << ",\n";
    json << "    \"mutual_information\": " << mutual_information << ",\n";
    json << "    \"adaptation_rate\": " << adaptation_rate << "\n";
    json << "  },\n";
    
    json << "  \"cortical_columns\": {\n";
    json << "    \"column_activity\": [";
    for (size_t i = 0; i < column_activity.size(); ++i) {
        if (i > 0) json << ", ";
        json << column_activity[i];
    }
    json << "],\n";
    json << "    \"column_frequencies\": [";
    for (size_t i = 0; i < column_frequencies.size(); ++i) {
        if (i > 0) json << ", ";
        json << column_frequencies[i];
    }
    json << "],\n";
    json << "    \"column_specialization\": [";
    for (size_t i = 0; i < column_specialization.size(); ++i) {
        if (i > 0) json << ", ";
        json << column_specialization[i];
    }
    json << "]\n";
    json << "  },\n";
    
    json << "  \"performance\": {\n";
    json << "    \"kernel_execution_time\": " << kernel_execution_time << ",\n";
    json << "    \"memory_transfer_time\": " << memory_transfer_time << ",\n";
    json << "    \"memory_usage_bytes\": " << memory_usage_bytes << ",\n";
    json << "    \"gpu_utilization\": " << gpu_utilization << ",\n";
    json << "    \"simulation_speed_factor\": " << simulation_speed_factor << ",\n";
    json << "    \"energy_consumption\": " << energy_consumption << "\n";
    json << "  },\n";
    
    json << "  \"stability\": {\n";
    json << "    \"numerical_stability\": " << numerical_stability << ",\n";
    json << "    \"simulation_stable\": " << (simulation_stable ? "true" : "false") << ",\n";
    json << "    \"cuda_error_count\": " << cuda_error_count << "\n";
    json << "  },\n";
    
    json << "  \"performance_history\": [";
    for (size_t i = 0; i < performance_history.size(); ++i) {
        if (i > 0) json << ", ";
        json << performance_history[i];
    }
    json << "],\n";
    
    json << "  \"analysis\": {\n";
    json << "    \"health_score\": " << getHealthScore() << ",\n";
    json << "    \"performance_score\": " << getPerformanceScore() << "\n";
    json << "  }\n";
    
    json << "}";
    
    return json.str();
}

bool NetworkStats::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    try {
        // Write binary header
        const char* header = "NEUROGENSTAT";
        file.write(header, 12);
        
        // Write version
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write core statistics structure
        file.write(reinterpret_cast<const char*>(this), sizeof(NetworkStats) - sizeof(std::vector<float>) * 4);
        
        // Write vector data separately
        auto writeVector = [&file](const std::vector<float>& vec) {
            uint32_t size = static_cast<uint32_t>(vec.size());
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            if (size > 0) {
                file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
            }
        };
        
        writeVector(column_activity);
        writeVector(column_frequencies);
        writeVector(column_specialization);
        writeVector(performance_history);
        
        return file.good();
    } catch (const std::exception&) {
        return false;
    }
}

bool NetworkStats::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    try {
        // Read and verify header
        char header[13] = {0};
        file.read(header, 12);
        if (std::string(header) != "NEUROGENSTAT") {
            return false;
        }
        
        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            return false; // Unsupported version
        }
        
        // Read core statistics structure
        file.read(reinterpret_cast<char*>(this), sizeof(NetworkStats) - sizeof(std::vector<float>) * 4);
        
        // Read vector data
        auto readVector = [&file](std::vector<float>& vec) {
            uint32_t size;
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            vec.resize(size);
            if (size > 0) {
                file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
            }
        };
        
        readVector(column_activity);
        readVector(column_frequencies);
        readVector(column_specialization);
        readVector(performance_history);
        
        return file.good();
    } catch (const std::exception&) {
        return false;
    }
}

// ============================================================================
// ADVANCED ANALYSIS FUNCTIONS
// ============================================================================

namespace NetworkAnalysis {

/**
 * Calculate network criticality based on branching ratio
 * Critical networks (branching ratio ≈ 1) show optimal information processing
 */
float calculateCriticality(const NetworkStats& stats) {
    if (stats.mean_firing_rate <= 0.0f || stats.total_synapses == 0) {
        return 0.0f;
    }
    
    // Estimate branching ratio from activity propagation
    float branching_ratio = (static_cast<float>(stats.active_synapses) / stats.total_synapses) * 
                           (stats.mean_firing_rate / 10.0f); // Normalize to expected range
    
    // Criticality is maximized when branching ratio ≈ 1
    return 1.0f - std::abs(branching_ratio - 1.0f);
}

/**
 * Estimate network entropy as a measure of information processing capacity
 */
float calculateNetworkEntropy(const NetworkStats& stats) {
    if (stats.neuron_activity_ratio <= 0.0f || stats.neuron_activity_ratio >= 1.0f) {
        return 0.0f;
    }
    
    float p = stats.neuron_activity_ratio;
    return -(p * std::log2(p) + (1.0f - p) * std::log2(1.0f - p));
}

/**
 * Assess metabolic efficiency of the network
 */
float calculateMetabolicEfficiency(const NetworkStats& stats) {
    if (stats.kernel_execution_time <= 0.0f) {
        return 0.0f;
    }
    
    // Efficiency = Information processed per unit energy (computational time)
    float information_rate = stats.network_entropy * stats.mean_firing_rate;
    float energy_cost = stats.kernel_execution_time; // Already in ms
    
    return information_rate / (energy_cost + 1e-6f); // Avoid division by zero
}

/**
 * Evaluate learning efficiency based on reward accumulation and plasticity
 */
float calculateLearningEfficiency(const NetworkStats& stats) {
    if (stats.simulation_steps <= 0) {
        return 0.0f;
    }
    
    // Learning efficiency = Cumulative reward per plasticity event
    float plasticity_events = stats.plasticity_rate * stats.simulation_steps;
    if (plasticity_events <= 0.0f) {
        return 0.0f;
    }
    
    return stats.cumulative_reward / plasticity_events;
}

} // namespace NetworkAnalysis

// ============================================================================
// REAL-TIME MONITORING UTILITIES
// ============================================================================

namespace NetworkMonitoring {

/**
 * Ring buffer for efficient performance tracking
 */
class PerformanceTracker {
private:
    std::vector<float> buffer_;
    size_t head_ = 0;
    size_t size_ = 0;
    size_t capacity_;
    
public:
    explicit PerformanceTracker(size_t capacity = 1000) : capacity_(capacity) {
        buffer_.resize(capacity);
    }
    
    void addSample(float value) {
        buffer_[head_] = value;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) size_++;
    }
    
    float getMean() const {
        if (size_ == 0) return 0.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < size_; ++i) {
            sum += buffer_[i];
        }
        return sum / size_;
    }
    
    float getStandardDeviation() const {
        if (size_ <= 1) return 0.0f;
        float mean = getMean();
        float variance = 0.0f;
        for (size_t i = 0; i < size_; ++i) {
            float diff = buffer_[i] - mean;
            variance += diff * diff;
        }
        return std::sqrt(variance / (size_ - 1));
    }
    
    float getTrend() const {
        if (size_ < 10) return 0.0f;
        
        // Simple linear regression to detect trend
        float x_mean = (size_ - 1) / 2.0f;
        float y_mean = getMean();
        
        float numerator = 0.0f, denominator = 0.0f;
        for (size_t i = 0; i < size_; ++i) {
            float x_diff = i - x_mean;
            float y_diff = buffer_[i] - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }
        
        return (denominator > 0.0f) ? numerator / denominator : 0.0f;
    }
};

/**
 * Anomaly detection for network health monitoring
 */
class AnomalyDetector {
private:
    PerformanceTracker firing_rate_tracker_;
    PerformanceTracker synchrony_tracker_;
    PerformanceTracker weight_tracker_;
    
public:
    AnomalyDetector() : firing_rate_tracker_(500), synchrony_tracker_(500), weight_tracker_(500) {}
    
    void update(const NetworkStats& stats) {
        firing_rate_tracker_.addSample(stats.mean_firing_rate);
        synchrony_tracker_.addSample(stats.network_synchrony);
        weight_tracker_.addSample(stats.mean_synaptic_weight);
    }
    
    struct AnomalyReport {
        bool firing_rate_anomaly = false;
        bool synchrony_anomaly = false;
        bool weight_anomaly = false;
        float anomaly_score = 0.0f;
        std::string description;
    };
    
    AnomalyReport detectAnomalies(const NetworkStats& current_stats) {
        AnomalyReport report;
        
        // Z-score based anomaly detection
        const float threshold = 3.0f; // 3-sigma rule
        
        // Firing rate anomaly
        float fr_mean = firing_rate_tracker_.getMean();
        float fr_std = firing_rate_tracker_.getStandardDeviation();
        if (fr_std > 0.0f) {
            float fr_zscore = std::abs(current_stats.mean_firing_rate - fr_mean) / fr_std;
            report.firing_rate_anomaly = (fr_zscore > threshold);
        }
        
        // Synchrony anomaly
        float sync_mean = synchrony_tracker_.getMean();
        float sync_std = synchrony_tracker_.getStandardDeviation();
        if (sync_std > 0.0f) {
            float sync_zscore = std::abs(current_stats.network_synchrony - sync_mean) / sync_std;
            report.synchrony_anomaly = (sync_zscore > threshold);
        }
        
        // Weight anomaly
        float weight_mean = weight_tracker_.getMean();
        float weight_std = weight_tracker_.getStandardDeviation();
        if (weight_std > 0.0f) {
            float weight_zscore = std::abs(current_stats.mean_synaptic_weight - weight_mean) / weight_std;
            report.weight_anomaly = (weight_zscore > threshold);
        }
        
        // Overall anomaly score
        int anomaly_count = report.firing_rate_anomaly + report.synchrony_anomaly + report.weight_anomaly;
        report.anomaly_score = static_cast<float>(anomaly_count) / 3.0f;
        
        // Generate description
        if (report.anomaly_score > 0.0f) {
            report.description = "Detected anomalies in: ";
            if (report.firing_rate_anomaly) report.description += "firing_rate ";
            if (report.synchrony_anomaly) report.description += "synchrony ";
            if (report.weight_anomaly) report.description += "weights ";
        } else {
            report.description = "Network operating normally";
        }
        
        return report;
    }
};

} // namespace NetworkMonitoring
