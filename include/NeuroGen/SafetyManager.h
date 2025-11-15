#ifndef SAFETY_MANAGER_H
#define SAFETY_MANAGER_H

#include <vector>
#include <string>
#include <queue>
#include <chrono>
#include "NeuroGen/Action.h"

struct Point {
    int x, y;
    Point(int x_, int y_) : x(x_), y(y_) {}
};

struct Rect {
    int x, y, width, height;
    Rect(int x_=0, int y_=0, int w_=0, int h_=0) : x(x_), y(y_), width(w_), height(h_) {}
    bool contains(const Point& p) const {
        return p.x >= x && p.x < x + width && p.y >= y && p.y < y + height;
    }
};

class SafetyManager {
public:
    static SafetyManager& getInstance();
    void enableGlobalSafety(bool enable);
    void setScreenBounds(int width, int height);
    void setMaxActionsPerSecond(int max_actions);
    bool checkRateLimit() const;
    bool checkSpatialBounds(int x, int y) const;
    bool isActionSafe(const BrowsingAction& action) const;
    void setScreenDimensions(int width, int height);
private:
    SafetyManager();
    bool global_safety_enabled_ = false;
    Rect screen_bounds_;
    std::vector<Rect> forbidden_regions_;
    int max_actions_per_second_ = 10;
    mutable std::queue<std::chrono::steady_clock::time_point> recent_actions_;
};

#endif // SAFETY_MANAGER_H
