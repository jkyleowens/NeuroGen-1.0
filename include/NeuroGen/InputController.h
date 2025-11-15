#ifndef INPUT_CONTROLLER_H
#define INPUT_CONTROLLER_H

#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>
#include <functional>
#include <string>

class InputController {
public:
    bool initialize();
    void shutdown();
    bool moveMouse(int x, int y);
    bool clickMouse(int x, int y, int button = 1);
    bool scrollMouse(int x, int y, int delta);
    bool typeText(const std::string& text);

    void enableSafetyBounds(int min_x, int min_y, int max_x, int max_y);
    void setEmergencyStop(std::function<bool()> stop_check);

private:
    bool isWithinSafetyBounds(int x, int y) const;
    Display* x11_display_ = nullptr;
    bool safety_enabled_ = false;
    struct Bounds { int min_x, min_y, max_x, max_y; } bounds_;
    std::function<bool()> emergency_stop_check_;
};

#endif // INPUT_CONTROLLER_H
