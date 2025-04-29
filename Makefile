# Makefile

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

# Paths
SRC_DIR = src
INCLUDE_DIR = $(SRC_DIR)/include
OBJ_DIR = build

# Files
TARGET = test
SRCS = $(SRC_DIR)/test.cpp $(SRC_DIR)/knncuda.cu

# Build target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(SRCS) -I$(INCLUDE_DIR) -o $(TARGET) $(NVCC_FLAGS)

clean:
	rm -f $(TARGET)
	