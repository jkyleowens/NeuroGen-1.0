#include "NeuroGen/InputController.h"
#include <iostream>
#include <unistd.h>
#include <cmath>

bool InputController::initialize() {
    x11_display_ = XOpenDisplay(nullptr);
    if (!x11_display_) {
        std::cerr << "InputController: Failed to open X display" << std::endl;
        return false;
    }
    return true;
}

void InputController::shutdown() {
    if (x11_display_) {
        XCloseDisplay(x11_display_);
        x11_display_ = nullptr;
    }
}

bool InputController::isWithinSafetyBounds(int x, int y) const {
    if (!safety_enabled_) return true;
    return x >= bounds_.min_x && x <= bounds_.max_x && y >= bounds_.min_y && y <= bounds_.max_y;
}

void InputController::enableSafetyBounds(int min_x, int min_y, int max_x, int max_y) {
    safety_enabled_ = true;
    bounds_ = {min_x, min_y, max_x, max_y};
}

void InputController::setEmergencyStop(std::function<bool()> stop_check) {
    emergency_stop_check_ = stop_check;
}

bool InputController::moveMouse(int x, int y) {
    if (!x11_display_ || !isWithinSafetyBounds(x, y)) return false;
    if (emergency_stop_check_ && emergency_stop_check_()) return false;
    XTestFakeMotionEvent(x11_display_, -1, x, y, CurrentTime);
    XFlush(x11_display_);
    return true;
}

bool InputController::clickMouse(int x, int y, int button) {
    if (!moveMouse(x, y)) return false;
    XTestFakeButtonEvent(x11_display_, button, True, CurrentTime);
    XTestFakeButtonEvent(x11_display_, button, False, CurrentTime);
    XFlush(x11_display_);
    return true;
}

bool InputController::scrollMouse(int x, int y, int delta) {
    if (!moveMouse(x, y)) return false;
    int button = (delta > 0) ? 4 : 5;
    int repeats = std::abs(delta);
    for (int i=0;i<repeats;++i) {
        XTestFakeButtonEvent(x11_display_, button, True, CurrentTime);
        XTestFakeButtonEvent(x11_display_, button, False, CurrentTime);
    }
    XFlush(x11_display_);
    return true;
}

bool InputController::typeText(const std::string& text) {
    if (!x11_display_) return false;
    for (char c : text) {
        KeySym keysym = XStringToKeysym(std::string(1,c).c_str());
        KeyCode keycode = XKeysymToKeycode(x11_display_, keysym);
        if (keycode == 0) continue;
        XTestFakeKeyEvent(x11_display_, keycode, True, CurrentTime);
        XTestFakeKeyEvent(x11_display_, keycode, False, CurrentTime);
    }
    XFlush(x11_display_);
    return true;
}
