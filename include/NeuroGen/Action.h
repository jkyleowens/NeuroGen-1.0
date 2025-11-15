#ifndef NEUROGEN_ACTION_H
#define NEUROGEN_ACTION_H

#include <string>
#include <vector>

// Enum for different types of browsing actions
enum class ActionType {
    CLICK,
    SCROLL,
    TYPE,
    ENTER,
    BACKSPACE,
    NAVIGATE, // For directly going to a URL
    WAIT,     // For pausing execution
    NONE      // No action
};

// Enum for scroll direction
enum class ScrollDirection {
    UP,
    DOWN
};

// Struct to represent a browsing action
struct BrowsingAction {
    ActionType type = ActionType::NONE;
    int x_coordinate = 0;
    int y_coordinate = 0;
    int scroll_amount = 0;
    ScrollDirection scroll_direction = ScrollDirection::DOWN;
    std::string text_content;
    std::string url;
    float confidence = 1.0f;
    int wait_duration_ms = 0; // Duration for WAIT action
};

#endif // NEUROGEN_ACTION_H
