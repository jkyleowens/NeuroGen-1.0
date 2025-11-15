# Compiler and Linker
CXX := clang++
NVCC := /opt/cuda/bin/nvcc
LINK := clang++

# Directories
SRC_DIR := src
OBJ_DIR := obj
BUILD_DIR := build
INCLUDE_DIR := include
CUDA_SRC_DIR := $(SRC_DIR)/cuda
CUDA_OBJ_DIR := $(OBJ_DIR)/cuda
DEPS_DIR := $(BUILD_DIR)/deps
CUDA_DEPS_DIR := $(DEPS_DIR)/cuda

# CUDA Path
CUDA_PATH := /opt/cuda

# Executable Name
TARGET := NeuroGen
TARGET_AUTONOMOUS := NeuroGen_Autonomous

# Compiler Flags
# Note: The -I$(INCLUDE_DIR) flag tells the compilers where to find your header files.
CXXFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -O3 -g -fPIC -Wall -ferror-limit=50
NVCCFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -arch=sm_75 -O3 -g -lineinfo \
             -Xcompiler -fPIC -Xcompiler -Wall -use_fast_math \
             --expt-relaxed-constexpr --expt-extended-lambda -ccbin /usr/bin/clang++

# Linker Flags
# Disabled CUDA libraries for CPU-only build
LDFLAGS := -L/usr/lib/x86_64-linux-gnu -L/usr/lib
# Link directly to .so.25 since .so symlink may not exist
LDLIBS := /usr/lib/x86_64-linux-gnu/libjsoncpp.so.25 -lX11

# --- Source Files ---

# Automatically find all .cpp files then exclude transitional/duplicate sources
ALL_CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Excluded sources causing duplicate symbol definitions or deprecated
# Also excluding CUDA-dependent sources when building without CUDA
EXCLUDE_SOURCES := \
    $(SRC_DIR)/NlpAgentImplementation.cpp \
    $(SRC_DIR)/execute_action_temp.cpp \
    $(SRC_DIR)/DecisionAndActionSystems_fixed.cpp \
    $(SRC_DIR)/EnhancedLearningSystem.cpp \
    $(SRC_DIR)/NetworkCUDA.cpp

# Separate main source files
MAIN_SRC := $(SRC_DIR)/main.cpp
AUTONOMOUS_MAIN_SRC := $(SRC_DIR)/main_autonomous.cpp

# Filter out excluded and autonomous main from default sources
CPP_SOURCES := $(filter-out $(EXCLUDE_SOURCES) $(AUTONOMOUS_MAIN_SRC), $(ALL_CPP_SOURCES))

# --- Object Files ---

# Generate object file names from source file names
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))

# Object for autonomous main (built separately)
AUTONOMOUS_MAIN_OBJECT := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(AUTONOMOUS_MAIN_SRC))

# Combine all object files
OBJECTS := $(CPP_OBJECTS)

# CUDA source files providing required wrapper symbols
CUDA_WRAPPER_SOURCES := \
    $(CUDA_SRC_DIR)/CudaKernelWrappers.cu \
    $(CUDA_SRC_DIR)/EnhancedSTDPKernel.cu \
    $(CUDA_SRC_DIR)/HebbianLearningKernel.cu \
    $(CUDA_SRC_DIR)/HomeostaticMechanismsKernel.cu \
    $(CUDA_SRC_DIR)/NeuromodulationKernels.cu

# CUDA objects
CUDA_WRAPPER_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_WRAPPER_SOURCES))

# Append CUDA wrapper objects to link line for targets needing EnhancedLearningSystem
# Disabled for CPU-only builds
# OBJECTS += $(CUDA_WRAPPER_OBJECTS)

# --- Dependency Files ---
DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$(CPP_SOURCES))

# --- Build Rules ---

all: $(TARGET)

autonomous: $(TARGET_AUTONOMOUS)

# Linking the final executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(LINK) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Linking the autonomous learning executable
$(TARGET_AUTONOMOUS): $(filter-out $(OBJ_DIR)/main.o,$(OBJECTS)) $(AUTONOMOUS_MAIN_OBJECT)
	@echo "Linking $(TARGET_AUTONOMOUS)..."
	$(LINK) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# C++ compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) $(DEPS_DIR)
	@echo "Compiling C++ source: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@$(CXX) $(CXXFLAGS) -MM $< -MT $@ -MF $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$<)

# CUDA compilation rule (for wrapper sources only for now)
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_OBJ_DIR)
	@echo "Compiling CUDA source: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- Directory Creation ---

# Create directories if they don't exist
$(OBJ_DIR) $(DEPS_DIR):
	mkdir -p $@

# Ensure CUDA object directory exists
$(CUDA_OBJ_DIR):
	mkdir -p $@

# --- Housekeeping ---

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(CUDA_OBJ_DIR) $(TARGET) $(TARGET_AUTONOMOUS) $(DEPS_DIR)

# Test targets
test_brain_architecture: test_brain_module_architecture.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

.PHONY: all autonomous clean test_brain_architecture

# Include dependency files
-include $(DEPS)
