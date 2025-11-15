#ifndef TASK_AUTOMATION_MODULES_H
#define TASK_AUTOMATION_MODULES_H

#include "NeuroGen/NeuralModule.h"
#include <iostream>
#include <memory>
#include <vector>

/**
 * @class TaskModule
 * @brief Abstract base class for all modules that perform a specific task.
 */
class TaskModule {
public:
    virtual ~TaskModule() = default;

    // A pure virtual function that all derived modules MUST implement.
    virtual void initialize() = 0;
};

/**
 * @class CognitiveModule
 * @brief A task module responsible for high-level cognitive processes.
 */
class CognitiveModule : public TaskModule {
public:
    CognitiveModule(std::shared_ptr<NeuralModule> perception, std::shared_ptr<NeuralModule> planning)
        : perception_module_(perception), planning_module_(planning) {}

    // >>> FIX: Changed the definition to a declaration by removing the function body.
    void initialize() override;
    // <<< END FIX

private:
    std::shared_ptr<NeuralModule> perception_module_;
    std::shared_ptr<NeuralModule> planning_module_;
};

/**
 * @class MotorModule
 * @brief A task module responsible for motor control outputs.
 */
class MotorModule : public TaskModule {
public:
    MotorModule(std::shared_ptr<NeuralModule> motor_control)
        : motor_control_module_(motor_control) {}

    // >>> FIX: Changed the definition to a declaration.
    void initialize() override;
    // <<< END FIX

private:
    std::shared_ptr<NeuralModule> motor_control_module_;
};

#endif // TASK_AUTOMATION_MODULES_H