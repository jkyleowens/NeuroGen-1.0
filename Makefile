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

# Python Module Name (shared library)
TARGET := neural_network.so
TARGET_AUTONOMOUS := NeuroGen_Autonomous

# Python configuration for pybind11
PYTHON_CONFIG := python3-config
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null)

# Compiler Flags
# Note: The -I$(INCLUDE_DIR) flag tells the compilers where to find your header files.
CXXFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES) -O3 -g -fPIC -Wall -ferror-limit=50
NVCCFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -arch=sm_75 -O3 -g -lineinfo \
             -Xcompiler -fPIC -Xcompiler -Wall -use_fast_math \
             --expt-relaxed-constexpr --expt-extended-lambda -ccbin /usr/bin/clang++

# Linker Flags for Python module (shared library)
# When CUDA is not available, comment out CUDA paths and libraries
# LDFLAGS := -L$(CUDA_PATH)/lib64 -L/usr/lib
# LDLIBS := -ljsoncpp -lcudart -lcurand -lcublas -lcufft -lX11 -lXtst

# CPU-only build (without CUDA) - Python module
LDFLAGS := -L/usr/lib $(PYTHON_LDFLAGS)
LDLIBS := -lX11

# --- Source Files ---

# Automatically find all .cpp files then exclude transitional/duplicate sources
ALL_CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Excluded sources causing duplicate symbol definitions, deprecated, or requiring CUDA
EXCLUDE_SOURCES := \
    $(SRC_DIR)/NlpAgentImplementation.cpp \
    $(SRC_DIR)/execute_action_temp.cpp \
    $(SRC_DIR)/DecisionAndActionSystems_fixed.cpp \
    $(SRC_DIR)/EnhancedLearningSystem.cpp \
    $(SRC_DIR)/NetworkCUDA.cpp

# Separate main source files (excluded from Python module build)
MAIN_SRC := $(SRC_DIR)/main.cpp
AUTONOMOUS_MAIN_SRC := $(SRC_DIR)/main_autonomous.cpp

# Filter out excluded sources and main files (Python module doesn't need main)
CPP_SOURCES := $(filter-out $(EXCLUDE_SOURCES) $(MAIN_SRC) $(AUTONOMOUS_MAIN_SRC), $(ALL_CPP_SOURCES))

# --- Object Files ---

# Generate object file names from source file names
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))

# Object for autonomous main (built separately)
AUTONOMOUS_MAIN_OBJECT := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(AUTONOMOUS_MAIN_SRC))

# Combine all object files
OBJECTS := $(CPP_OBJECTS)

# CUDA source files providing required wrapper symbols and core network functionality
# To enable CUDA: uncomment these lines when CUDA toolkit is installed
# CUDA_WRAPPER_SOURCES := \
#     $(CUDA_SRC_DIR)/CudaKernelWrappers.cu \
#     $(CUDA_SRC_DIR)/EnhancedSTDPKernel.cu \
#     $(CUDA_SRC_DIR)/HebbianLearningKernel.cu \
#     $(CUDA_SRC_DIR)/HomeostaticMechanismsKernel.cu \
#     $(CUDA_SRC_DIR)/NeuromodulationKernels.cu \
#     $(CUDA_SRC_DIR)/LearningStateKernels.cu \
#     $(CUDA_SRC_DIR)/KernelLaunchWrappers.cu \
#     $(CUDA_SRC_DIR)/NetworkCUDA.cu \
#     $(CUDA_SRC_DIR)/NetworkCUDA_Interface.cu

# CUDA objects
# CUDA_WRAPPER_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_WRAPPER_SOURCES))

# Append CUDA wrapper objects to link line for targets needing EnhancedLearningSystem
# OBJECTS += $(CUDA_WRAPPER_OBJECTS)

# --- Dependency Files ---
DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$(CPP_SOURCES))

# --- Build Rules ---

all: $(TARGET)

autonomous: $(TARGET_AUTONOMOUS)

# Linking the Python module (shared library)
$(TARGET): $(OBJECTS)
	@echo "Linking Python module $(TARGET)..."
	$(LINK) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

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
