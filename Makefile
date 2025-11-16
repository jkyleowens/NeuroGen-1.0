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

PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)

# Executable and Module Names
TARGET_SO := neural_network.so
TARGET_NEUROGEN := NeuroGen
TARGET_AUTONOMOUS := NeuroGen_Autonomous
 
# Python configuration for pybind11
PYTHON_CONFIG := python3-config
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null)
 
# Compiler Flags
CXXFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES) -O3 -g -fPIC -Wall -ferror-limit=50
NVCCFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -arch=sm_75 -O3 -g -lineinfo \
             -Xcompiler -fPIC -Xcompiler -Wall -use_fast_math \
             --expt-relaxed-constexpr --expt-extended-lambda -ccbin /usr/bin/clang++
 
# Linker Flags - WITH CUDA for executables
LDFLAGS_CUDA := -L$(CUDA_PATH)/lib64 -L/usr/lib
LDLIBS_CUDA := -lcudart -lcurand -lcublas -lX11 -lXtst

# Linker Flags - For Python module (needs Python libs)
LDFLAGS_PYTHON := -L$(CUDA_PATH)/lib64 -L/usr/lib $(PYTHON_LDFLAGS)
LDLIBS_PYTHON := -lcudart -lcurand -lcublas -lX11 -lXtst

# --- Source Files ---

# Automatically find all .cpp files then exclude transitional/duplicate sources
ALL_CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Excluded sources causing duplicate symbol definitions or deprecated
EXCLUDE_SOURCES := \
    $(SRC_DIR)/NlpAgentImplementation.cpp \
    $(SRC_DIR)/execute_action_temp.cpp \
    $(SRC_DIR)/DecisionAndActionSystems_fixed.cpp

# Separate main source files
MAIN_SRC := $(SRC_DIR)/main.cpp
AUTONOMOUS_MAIN_SRC := $(SRC_DIR)/main_autonomous.cpp
BINDINGS_SRC := $(SRC_DIR)/bindings.cpp

# Filter out excluded sources, main files, and bindings (bindings only for Python module)
CPP_SOURCES := $(filter-out $(EXCLUDE_SOURCES) $(MAIN_SRC) $(AUTONOMOUS_MAIN_SRC) $(BINDINGS_SRC), $(ALL_CPP_SOURCES))

# --- Object Files ---

# Generate object file names from source file names
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))

# Objects for main executables
MAIN_OBJECT := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_SRC))
AUTONOMOUS_MAIN_OBJECT := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(AUTONOMOUS_MAIN_SRC))
BINDINGS_OBJECT := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(BINDINGS_SRC))

# CUDA source files
CUDA_WRAPPER_SOURCES := \
    $(CUDA_SRC_DIR)/CudaKernelWrappers.cu \
    $(CUDA_SRC_DIR)/EnhancedSTDPKernel.cu \
    $(CUDA_SRC_DIR)/HebbianLearningKernel.cu \
    $(CUDA_SRC_DIR)/HomeostaticMechanismsKernel.cu \
    $(CUDA_SRC_DIR)/NeuromodulationKernels.cu

# CUDA objects
CUDA_WRAPPER_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_WRAPPER_SOURCES))

# Core objects (without any main or bindings) - used by executables
CORE_OBJECTS := $(CPP_OBJECTS) $(CUDA_WRAPPER_OBJECTS)

# Python module objects (core + bindings)
PYTHON_MODULE_OBJECTS := $(CORE_OBJECTS) $(BINDINGS_OBJECT)

# --- Dependency Files ---
DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$(CPP_SOURCES))

# --- Build Rules ---

all: $(TARGET_SO) $(TARGET_NEUROGEN) $(TARGET_AUTONOMOUS)

# Python module only
module: $(TARGET_SO)

# Executables only
executables: $(TARGET_NEUROGEN) $(TARGET_AUTONOMOUS)

# Linking the Python module (shared library)
$(TARGET_SO): $(PYTHON_MODULE_OBJECTS)
	@echo "Linking Python module $(TARGET_SO)..."
	$(LINK) -shared -o $@ $^ $(LDFLAGS_PYTHON) $(LDLIBS_PYTHON)

# Linking the main NeuroGen executable
$(TARGET_NEUROGEN): $(CORE_OBJECTS) $(MAIN_OBJECT)
	@echo "Linking $(TARGET_NEUROGEN) executable..."
	$(LINK) -o $@ $^ $(LDFLAGS_CUDA) $(LDLIBS_CUDA)

# Linking the autonomous learning executable
$(TARGET_AUTONOMOUS): $(CORE_OBJECTS) $(AUTONOMOUS_MAIN_OBJECT)
	@echo "Linking $(TARGET_AUTONOMOUS) executable..."
	$(LINK) -o $@ $^ $(LDFLAGS_CUDA) $(LDLIBS_CUDA)

# C++ compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) $(DEPS_DIR)
	@echo "Compiling C++ source: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@$(CXX) $(CXXFLAGS) -MM $< -MT $@ -MF $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$<)

# CUDA compilation rule
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
	rm -rf $(OBJ_DIR) $(CUDA_OBJ_DIR) $(TARGET_SO) $(TARGET_NEUROGEN) $(TARGET_AUTONOMOUS) $(DEPS_DIR)

# Test targets
test_brain_architecture: test_brain_module_architecture.cpp $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_CUDA) $(LDLIBS_CUDA)

.PHONY: all module executables clean test_brain_architecture

# Include dependency files
-include $(DEPS)