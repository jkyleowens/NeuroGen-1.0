// ============================================================================
// VISUAL INTERFACE HEADER
// File: include/NeuroGen/VisualInterface.h
// ============================================================================

#ifndef VISUAL_INTERFACE_H
#define VISUAL_INTERFACE_H

// #include <vector>
// #include <string>
// #include <memory>
// #include <opencv2/core.hpp>
// #include "RealScreenCapture.h"
// #include "GUIElementDetector.h"
// #include "OCRProcessor.h"
// #include "BioVisualProcessor.h"
// #include "CommonStructs.h"

// class AutonomousLearningAgent; // Forward declaration

// class VisualInterface {
// public:
//     VisualInterface(bool use_real_screen = false);
//     ~VisualInterface();

//     void initialize(std::shared_ptr<AutonomousLearningAgent> agent);
//     std::vector<float> capture_and_process_screen();
//     std::vector<ScreenElement> detect_screen_elements();
//     ScreenElement find_element_by_type(const std::string& type) const;
//     bool is_element_visible(const ScreenElement& element) const;
//     void update_attention_map(const std::vector<float>& attention_weights);
//     std::vector<float> get_visual_features(const ScreenElement& element) const;
//     cv::Mat get_last_frame() const;
//     cv::Mat get_attention_map() const;

//     // For simulated environment
//     void set_simulated_screen(const cv::Mat& screen_image);

// private:
//     void process_frame(const cv::Mat& frame);
//     void update_detected_elements(const std::vector<ScreenElement>& elements);
//     std::vector<float> extract_features_from_roi(const cv::Mat& roi);

//     bool use_real_screen_;
//     std::vector<ScreenElement> detected_elements_;
//     std::string last_ocr_text_;
//     float visual_complexity_;
//     cv::Mat current_screen_;
//     cv::Mat attention_map_;

//     std::unique_ptr<RealScreenCapture> real_screen_capture_;
//     std::unique_ptr<GUIElementDetector> gui_detector_;
//     std::unique_ptr<OCRProcessor> ocr_processor_;
//     std::unique_ptr<BioVisualProcessor> visual_processor_;

//     // For inter-agent communication
//     std::weak_ptr<AutonomousLearningAgent> agent_; // Use weak_ptr to avoid circular dependency
// };

#endif // VISUAL_INTERFACE_H