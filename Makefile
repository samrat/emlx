# Private configuration
PRIV_DIR = $(MIX_APP_PATH)/priv
BUILD_DIR = cache/emlx-$(EMLX_VERSION)$(MLX_VARIANT)/objs
EMLX_SO = $(PRIV_DIR)/libemlx.so
EMLX_LIB_DIR = $(PRIV_DIR)/mlx/lib

# Build flags
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(MLX_INCLUDE_DIR) -Wall \
         -std=c++17
ifeq ($(LIBMLX_ENABLE_DEBUG),true)
		CFLAGS += -g
else
		CFLAGS += -O3
endif

LDFLAGS = -L$(MLX_LIB_DIR) -lmlx -shared

# Platform-specific settings
UNAME_S = $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    LDFLAGS += -flat_namespace -undefined dynamic_lookup -rpath @loader_path/mlx/lib
else
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
endif

# Source files
SOURCES = c_src/emlx_nif.cpp
OBJECTS = $(patsubst c_src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# Main targets
all: $(EMLX_SO)

$(PRIV_DIR):
	@ mkdir -p $(PRIV_DIR)

$(BUILD_DIR)/%.o: c_src/%.cpp
	@ mkdir -p $(BUILD_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@

$(EMLX_SO): $(PRIV_DIR) $(OBJECTS)
	@ echo "Copying MLX library to $(EMLX_LIB_DIR)"
	@ mkdir -p $(EMLX_LIB_DIR)
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(MLX_LIB_DIR)/* $(EMLX_LIB_DIR) ; \
	else \
		rm -rf $(EMLX_LIB_DIR) ; \
		ln -sf $(MLX_LIB_DIR) $(EMLX_LIB_DIR) ; \
	fi
	$(CXX) $(OBJECTS) -o $(EMLX_SO) $(LDFLAGS)

clean:
	rm -rf $(PRIV_DIR)
	rm -rf $(BUILD_DIR)

.PHONY: all clean
