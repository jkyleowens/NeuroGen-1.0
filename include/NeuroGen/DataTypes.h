#ifndef NEUROGEN_DATATYPES_H
#define NEUROGEN_DATATYPES_H

#include <vector>
#include <string>
#include <chrono>

// Forward declaration
struct BrowsingAction;

struct Episode {
    std::vector<float> state;
    BrowsingAction* action;
    float reward;
    std::vector<float> next_state;
    bool terminal;
    std::chrono::system_clock::time_point timestamp;
};

#endif // NEUROGEN_DATATYPES_H
