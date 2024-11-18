#include "erl_nif.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"
#include <map>
#include <string>
#include <numeric>

using namespace mlx::core;

std::map<const std::string, const mlx::core::Dtype> dtypes = {
    {"bool", mlx::core::bool_},
    {"uint8", mlx::core::uint8},
    {"uint16", mlx::core::uint16},
    {"uint32", mlx::core::uint32},
    {"uint64", mlx::core::uint64},
    {"int8", mlx::core::int8},
    {"int16", mlx::core::int16},
    {"int32", mlx::core::int32},
    {"int64", mlx::core::int64},
    {"float16", mlx::core::float16},
    {"float32", mlx::core::float32},
    {"bfloat16", mlx::core::bfloat16},
    {"complex64", mlx::core::complex64}
};

std::map<const std::string, const uint8_t> dtype_sizes = {
    {"bool", mlx::core::bool_.size()},
    {"uint8", mlx::core::uint8.size()},
    {"uint16", mlx::core::uint16.size()},
    {"uint32", mlx::core::uint32.size()},
    {"uint64", mlx::core::uint64.size()},
    {"int8", mlx::core::int8.size()},
    {"int16", mlx::core::int16.size()},
    {"int32", mlx::core::int32.size()},
    {"int64", mlx::core::int64.size()},
    {"float16", mlx::core::float16.size()},
    {"float32", mlx::core::float32.size()},
    {"bfloat16", mlx::core::bfloat16.size()},
    {"complex64", mlx::core::complex64.size()}
};

inline mlx::core::Dtype string2dtype(const std::string &atom) {
    auto it = dtypes.find(atom);
    if (it != dtypes.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown dtype: " + atom);
}

inline const std::string *dtype2string(const mlx::core::Dtype dtype) {
    for (const auto& pair : dtypes) {
        if (pair.second == dtype) {
            return &pair.first;
        }
    }
    return nullptr;
}

// Class to manage the refcount of MLX arrays
class ArrayP {
 public:
  ArrayP(ErlNifEnv *env, const ERL_NIF_TERM arg) : ptr(nullptr) {
    // setup
    if (!enif_get_resource(env, arg, ARRAY_TYPE, (void **)&ptr)) {
      err = nx::nif::error(env, "Unable to get array param in NIF");
      return;
    }

    refcount = (std::atomic<int> *)(ptr + 1);
    deleted = (std::atomic_flag *)(refcount + 1);

    if (refcount->load() == 0) {
      // already deallocated
      ptr = nullptr;
      err = nx::nif::error(env, "Array has been deallocated");
      return;
    }

    if (is_valid()) {
      // increase reference count
      ++(*refcount);
    }
  }

  ~ArrayP() {
    if (is_valid()) {
      // decrease reference count
      if (refcount->fetch_sub(1) == 0) {
        ptr->~array();  // Call MLX array destructor
      }
    }
  }

  bool deallocate() {
    if (is_valid() && atomic_flag_test_and_set(deleted) == false) {
      --(*refcount);
      return true;
    } else {
      return false;
    }
  }

  mlx::core::array *data() const {
    return ptr;
  }

  bool is_valid() const {
    return ptr != nullptr;
  }

  ERL_NIF_TERM error() {
    return err;
  }

 private:
  mlx::core::array *ptr;
  std::atomic<int> *refcount;
  std::atomic_flag *deleted;
  ERL_NIF_TERM err;
};

#define CATCH()                                                  \
  catch (const std::exception& e) {                             \
    std::ostringstream msg;                                     \
    msg << e.what() << " in NIF." << __func__ << "/" << argc;  \
    return nx::nif::error(env, msg.str().c_str());             \
  }                                                             \
  catch (...) {                                                 \
    return nx::nif::error(env, "Unknown error occurred");       \
  }

#define ARRAY(A)                                            \
  try {                                                      \
    return nx::nif::ok(env, create_array_resource(env, A)); \
  }                                                          \
  CATCH()

ERL_NIF_TERM
create_array_resource(ErlNifEnv *env, mlx::core::array array) {
  ERL_NIF_TERM ret;
  mlx::core::array *arrayPtr;
  std::atomic<int> *refcount;

  arrayPtr = (mlx::core::array *)enif_alloc_resource(ARRAY_TYPE, sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
  if (arrayPtr == NULL)
    return enif_make_badarg(env);

  new (arrayPtr) mlx::core::array(std::move(array));
  refcount = new (arrayPtr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, arrayPtr);
  enif_release_resource(arrayPtr);

  return ret;
}

#define NIF(NAME) ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define PARAM(ARGN, TYPE, VAR) \
  TYPE VAR;                    \
  GET(ARGN, VAR)

#define ARRAY_PARAM(ARGN, VAR)      \
  ArrayP VAR##_tp(env, argv[ARGN]); \
  mlx::core::array *VAR;                \
  if (!VAR##_tp.is_valid()) {        \
    return VAR##_tp.error();         \
  } else {                           \
    VAR = VAR##_tp.data();           \
  }

#define LIST_PARAM(ARGN, TYPE, VAR)             \
TYPE VAR;                                      \
if (!nx::nif::get_list(env, argv[ARGN], VAR)) \
return nx::nif::error(env, "Unable to get " #VAR " list param.");

inline mlx::core::array make_scalar_tensor(double value, const mlx::core::Dtype& dtype) {
    return mlx::core::array(value, dtype);
}

NIF(scalar_type) {
  ARRAY_PARAM(0, t);

  const std::string *type_name = dtype2string(t->dtype());

  if (type_name != nullptr)
    return nx::nif::ok(env, enif_make_atom(env, type_name->c_str()));
  else
    return nx::nif::error(env, "Could not determine array type.");
}

NIF(make_zeros) {
    if (argc != 1) {
        return enif_make_badarg(env);
    }

    unsigned int length;
    if (!enif_get_list_length(env, argv[0], &length)) {
        return enif_make_badarg(env);
    }

    std::vector<int> shape;
    ERL_NIF_TERM head, tail = argv[0];
    
    for (unsigned int i = 0; i < length; i++) {
        int dim;
        if (!enif_get_list_cell(env, tail, &head, &tail) ||
            !enif_get_int(env, head, &dim)) {
            return enif_make_badarg(env);
        }
        shape.push_back(dim);
    }

    try {
        // Create MLX array filled with zeros
        mlx::core::array result = mlx::core::zeros(shape, mlx::core::float32);
        
        // Allocate resource for the array
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        // Copy the array into the resource
        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        // Create Erlang term
        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    } catch (...) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_atom(env, "unknown_error"));
    }
}

NIF(make_ones) {
    if (argc != 1) {
        return enif_make_badarg(env);
    }

    unsigned int length;
    if (!enif_get_list_length(env, argv[0], &length)) {
        return enif_make_badarg(env);
    }

    std::vector<int> shape;
    ERL_NIF_TERM head, tail = argv[0];
    
    for (unsigned int i = 0; i < length; i++) {
        int dim;
        if (!enif_get_list_cell(env, tail, &head, &tail) ||
            !enif_get_int(env, head, &dim)) {
            return enif_make_badarg(env);
        }
        shape.push_back(dim);
    }

    try {
        // Create MLX array filled with ones
        mlx::core::array result = mlx::core::ones(shape, mlx::core::float32);
        
        // Allocate resource for the array
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        // Copy the array into the resource
        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        // Create Erlang term
        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    } catch (...) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_atom(env, "unknown_error"));
    }
}

NIF(sum) {
    ARRAY_PARAM(0, t);  
    
    // Get the axes vector
    unsigned int length;
    std::vector<int> axes;
    if (!enif_get_list_length(env, argv[1], &length)) {
        return enif_make_badarg(env);
    }
    
    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < length; i++) {
        int axis;
        if (!enif_get_list_cell(env, tail, &head, &tail) ||
            !enif_get_int(env, head, &axis)) {
            return enif_make_badarg(env);
        }
        axes.push_back(axis);
    }

    // Get keepdims parameter
    int keep_dims;
    if (!enif_get_int(env, argv[2], &keep_dims)) {
        return enif_make_badarg(env);
    }

    try {
        // Create MLX array result
        mlx::core::array result = mlx::core::sum(*t, axes, static_cast<bool>(keep_dims));
        
        // Allocate resource for the array
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        // Copy the array into the resource
        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        // Create Erlang term
        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    } catch (...) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_atom(env, "unknown_error"));
    }
}

NIF(shape) {
  ARRAY_PARAM(0, t);

  std::vector<ERL_NIF_TERM> sizes;
  for (int64_t dim = 0; dim < t->ndim(); dim++)
    sizes.push_back(nx::nif::make(env, static_cast<int64_t>(t->shape()[dim])));

  return nx::nif::ok(env, enif_make_tuple_from_array(env, sizes.data(), sizes.size()));
}

NIF(to_type) {
    ARRAY_PARAM(0, t);
    
    char type_str[32];
    if (!enif_get_atom(env, argv[1], type_str, sizeof(type_str), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }

    try {
        mlx::core::Dtype new_dtype = string2dtype(type_str);
        mlx::core::array result = mlx::core::astype(*t, new_dtype);
        
        // Allocate and return new array resource
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

NIF(to_blob) {
  ARRAY_PARAM(0, t);
  
  try {
    // Evaluate the array to ensure data is available
    mlx::core::eval(*t);
    
    size_t byte_size = t->nbytes();
    int64_t limit = 0;

    bool has_received_limit = (argc == 2);

    if (has_received_limit) {
      PARAM(1, int64_t, param_limit);
      limit = param_limit;
      byte_size = limit * t->itemsize();
    }

    ERL_NIF_TERM result;
    void* result_data = (void*)enif_make_new_binary(env, byte_size, &result);

    // Get raw pointer to data and copy
    const void* src_data = t->data<void>();
    if (src_data == nullptr) {
      return nx::nif::error(env, "Failed to get array data");
    }
    std::memcpy(result_data, src_data, byte_size);
    
    return nx::nif::ok(env, result);
  } catch (const std::exception& e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env, "Unknown error during data copy");
  }
}

uint64_t elem_count(std::vector<int> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

NIF(from_blob) {
  BINARY_PARAM(0, blob);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env, "Binary size is too small for the requested shape");

  try {
    // Create MLX array directly from the binary data
    mlx::core::array array(blob.data, shape, type);

    ARRAY(array);
  } catch (const std::exception& e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env, "Unknown error creating array from binary data");
  }
}


NIF(scalar_tensor) {
    if (argc != 2) {
        return enif_make_badarg(env);
    }

    // Get scalar value
    double value;
    if (!enif_get_double(env, argv[0], &value)) {
        return enif_make_badarg(env);
    }

    // Get type
    char type_str[32];
    if (!enif_get_atom(env, argv[1], type_str, sizeof(type_str), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }

    try {
        mlx::core::Dtype dtype = string2dtype(type_str);
        mlx::core::array result = make_scalar_tensor(value, dtype);
        
        // Allocate resource for the array
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        // Copy the array into the resource
        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        // Create Erlang term
        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

static ErlNifFunc nif_funcs[] = {
    {"zeros", 1, make_zeros},
    {"ones", 1, make_ones},
    {"scalar_type", 1, scalar_type},
    {"sum", 3, sum},
    {"shape", 1, shape},
    {"to_type", 2, to_type},
    {"to_blob", 2, to_blob},
    {"from_blob", 3, from_blob},
    {"scalar_tensor", 2, scalar_tensor},
};

static void free_array(ErlNifEnv* env, void* obj) {
    mlx::core::array* arr = static_cast<mlx::core::array*>(obj);
    if (arr != nullptr) {
        arr->~array();
    }
}

static int open_resource_type(ErlNifEnv* env) {
    const char* name = "MLXArray";
    ErlNifResourceFlags flags = (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

    ARRAY_TYPE = enif_open_resource_type(env, NULL, name, free_array, flags, NULL);
    if (ARRAY_TYPE == NULL) {
        return -1;
    }
    return 0;
}

// In your module load function:
static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    if (open_resource_type(env) != 0) {
        return -1;
    }
    return 0;
}

// Update the NIF initialization
ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, NULL, NULL)
