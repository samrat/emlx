#include "erl_nif.h"
#include "mlx/backend/common/utils.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"

#include <map>
#include <numeric>
#include <string>

using namespace mlx::core;

std::map<const std::string, const mlx::core::Dtype> dtypes = {
    {"bool", mlx::core::bool_},         {"uint8", mlx::core::uint8},
    {"uint16", mlx::core::uint16},      {"uint32", mlx::core::uint32},
    {"uint64", mlx::core::uint64},      {"int8", mlx::core::int8},
    {"int16", mlx::core::int16},        {"int32", mlx::core::int32},
    {"int64", mlx::core::int64},        {"float16", mlx::core::float16},
    {"float32", mlx::core::float32},    {"bfloat16", mlx::core::bfloat16},
    {"complex64", mlx::core::complex64}};

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
    {"complex64", mlx::core::complex64.size()}};

inline mlx::core::Dtype string2dtype(const std::string &atom) {
  auto it = dtypes.find(atom);
  if (it != dtypes.end()) {
    return it->second;
  }
  throw std::runtime_error("Unknown dtype: " + atom);
}

inline const std::string *dtype2string(const mlx::core::Dtype dtype) {
  for (const auto &pair : dtypes) {
    if (pair.second == dtype) {
      return &pair.first;
    }
  }
  return nullptr;
}

inline const mlx::core::Device string2device(const std::string &atom) {
  if (atom == "cpu") {
    return mlx::core::Device(mlx::core::Device::DeviceType::cpu, 0);
  } else if (atom == "gpu") {
    return mlx::core::Device(mlx::core::Device::DeviceType::gpu, 0);
  }
  throw std::runtime_error("Unknown device: " + atom);
}

// Class to manage the refcount of MLX tensors
class TensorP {
public:
  TensorP(ErlNifEnv *env, const ERL_NIF_TERM arg) : ptr(nullptr) {
    // setup
    if (!enif_get_resource(env, arg, TENSOR_TYPE, (void **)&ptr)) {
      err = nx::nif::error(env, "Unable to get tensor param in NIF");
      return;
    }

    refcount = (std::atomic<int> *)(ptr + 1);
    deleted = (std::atomic_flag *)(refcount + 1);

    if (refcount->load() == 0) {
      // already deallocated
      ptr = nullptr;
      err = nx::nif::error(env, "Tensor has been deallocated");
      return;
    }

    if (is_valid()) {
      // increase reference count
      ++(*refcount);
    }
  }

  ~TensorP() {
    if (is_valid()) {
      // decrease reference count
      if (refcount->fetch_sub(1) == 0) {
        ptr->~array(); // Call MLX tensor destructor
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

  mlx::core::array *data() const { return ptr; }

  bool is_valid() const { return ptr != nullptr; }

  ERL_NIF_TERM error() { return err; }

private:
  mlx::core::array *ptr;
  std::atomic<int> *refcount;
  std::atomic_flag *deleted;
  ERL_NIF_TERM err;
};

#define CATCH()                                                                \
  catch (const std::exception &e) {                                            \
    std::ostringstream msg;                                                    \
    msg << e.what() << " in NIF." << __func__ << "/" << argc;                  \
    return nx::nif::error(env, msg.str().c_str());                             \
  }                                                                            \
  catch (...) {                                                                \
    return nx::nif::error(env, "Unknown error occurred");                      \
  }

#define TENSOR(A)                                                              \
  try {                                                                        \
    return nx::nif::ok(env, create_tensor_resource(env, A));                   \
  }                                                                            \
  CATCH()

ERL_NIF_TERM
create_tensor_resource(ErlNifEnv *env, mlx::core::array tensor) {
  ERL_NIF_TERM ret;
  mlx::core::array *tensorPtr;
  std::atomic<int> *refcount;

  tensorPtr = (mlx::core::array *)enif_alloc_resource(
      TENSOR_TYPE, sizeof(mlx::core::array) + sizeof(std::atomic<int>) +
                       sizeof(std::atomic_flag));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) mlx::core::array(std::move(tensor));
  refcount = new (tensorPtr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, tensorPtr);
  enif_release_resource(tensorPtr);

  return ret;
}

#define NIF(NAME)                                                              \
  ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define PARAM(ARGN, TYPE, VAR)                                                 \
  TYPE VAR;                                                                    \
  GET(ARGN, VAR)

#define TENSOR_PARAM(ARGN, VAR)                                                \
  TensorP VAR##_tp(env, argv[ARGN]);                                           \
  mlx::core::array *VAR;                                                       \
  if (!VAR##_tp.is_valid()) {                                                  \
    return VAR##_tp.error();                                                   \
  } else {                                                                     \
    VAR = VAR##_tp.data();                                                     \
  }

#define LIST_PARAM(ARGN, TYPE, VAR)                                            \
  TYPE VAR;                                                                    \
  if (!nx::nif::get_list(env, argv[ARGN], VAR))                                \
    return nx::nif::error(env, "Unable to get " #VAR " list param.");

NIF(deallocate) {
  TensorP t(env, argv[0]);
  if (t.deallocate()) {
    return nx::nif::ok(env);
  } else {
    return nx::nif::atom(env, "already_deallocated");
  }
}

NIF(scalar_type) {
  TENSOR_PARAM(0, t);

  const std::string *type_name = dtype2string(t->dtype());

  if (type_name != nullptr)
    return nx::nif::ok(env, enif_make_atom(env, type_name->c_str()));
  else
    return nx::nif::error(env, "Could not determine tensor type.");
}

NIF(all) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_dims);
  DEVICE_PARAM(3, device);

  if (axes.empty()) {
    for (int i = 0; i < t->ndim(); ++i) {
      axes.push_back(i);
    }
  }

  auto result = mlx::core::all(*t, axes, keep_dims, device);

  TENSOR(result);
}

NIF(any) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_dims);
  DEVICE_PARAM(3, device);

  if (axes.empty()) {
    for (int i = 0; i < t->ndim(); ++i) {
      axes.push_back(i);
    }
  }

  auto result = mlx::core::any(*t, axes, keep_dims, device);

  TENSOR(result);
}

NIF(sum) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_dims);
  DEVICE_PARAM(3, device);

  // If axes is empty, sum over all dimensions
  // MLX sums over all dimensions ONLY if axes is not specified.
  // If an empty vector is passed, it will not sum over any dimensions.
  if (axes.empty()) {
    for (int i = 0; i < t->ndim(); ++i) {
      axes.push_back(i);
    }
  }

  auto result = mlx::core::sum(*t, axes, keep_dims, device);

  TENSOR(result);
}

NIF(product) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_dims);
  DEVICE_PARAM(3, device);

  if (axes.empty()) {
    for (int i = 0; i < t->ndim(); ++i) {
      axes.push_back(i);
    }
  }

  auto result = mlx::core::prod(*t, axes, keep_dims, device);

  TENSOR(result);
}

NIF(shape) {
  TENSOR_PARAM(0, t);

  std::vector<ERL_NIF_TERM> sizes;
  for (int64_t dim = 0; dim < t->ndim(); dim++)
    sizes.push_back(nx::nif::make(env, static_cast<int64_t>(t->shape()[dim])));

  return nx::nif::ok(
      env, enif_make_tuple_from_array(env, sizes.data(), sizes.size()));
}

NIF(ones) {
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::ones(shape, type, device));
}

NIF(zeros) {
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::zeros(shape, type, device));
}

NIF(reshape) {
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::reshape(*t, shape, device));
}

NIF(to_type) {
  TENSOR_PARAM(0, t);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::astype(*t, type, device));
}

NIF(to_blob) {
  ERL_NIF_TERM result;
  TENSOR_PARAM(0, t);

  size_t byte_size = t->nbytes();
  int limit = 0;
  bool has_received_limit = (argc == 2);

  if (has_received_limit) {
    PARAM(1, int, param_limit);
    limit = param_limit;
    byte_size = limit * t->itemsize();
  }

  // Flatten and slice if needed
  mlx::core::array flattened = mlx::core::flatten(*t);
  mlx::core::array reshaped =
      (has_received_limit && byte_size < t->nbytes())
          ? mlx::core::slice(flattened, std::vector<int>{0},
                             std::vector<int>{limit})
          : flattened;

  // Evaluate to ensure data is available
  mlx::core::eval(reshaped);

  // Create result binary
  void *result_data = (void *)enif_make_new_binary(env, byte_size, &result);

  // The MLX array data may not be contiguous in memory, even after the
  // reshape+flatten operations. See:
  // https://github.com/ml-explore/mlx/discussions/1608#discussioncomment-11332071
  //
  // Set up contiguous iterator
  std::vector<int> slice_sizes(reshaped.shape().begin(),
                               reshaped.shape().end());
  ContiguousIterator<size_t> iterator(slice_sizes, reshaped.strides(),
                                      reshaped.ndim());

  // Copy data element by element using iterator
  size_t element_size = reshaped.itemsize();
  const char *src_data = static_cast<const char *>(reshaped.data<void>());
  char *dst_data = static_cast<char *>(result_data);

  size_t num_elements = byte_size / element_size;
  for (size_t i = 0; i < num_elements; i++) {
    size_t src_offset = iterator.loc;
    std::memcpy(dst_data + (i * element_size),
                src_data + (src_offset * element_size), element_size);
    iterator.step();
  }

  return nx::nif::ok(env, result);
}

uint64_t elem_count(std::vector<int> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

NIF(from_blob) {
  BINARY_PARAM(0, blob);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);
  // DEVICE_PARAM(3, device);

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env,
                          "Binary size is too small for the requested shape");

  try {
    // Allocate MLX buffer and copy data from blob
    size_t byte_size = blob.size;
    allocator::Buffer mlx_buf = allocator::malloc(byte_size);
    void *buf_ptr = mlx_buf.raw_ptr();

    // Copy binary data to MLX buffer
    std::memcpy(buf_ptr, blob.data, byte_size);

    // Create deleter for the buffer
    auto deleter = [](allocator::Buffer buf) { allocator::free(buf); };

    // Create MLX array from the buffer
    TENSOR(mlx::core::array(mlx_buf, shape, type, deleter));
  } catch (const std::exception &e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env,
                          "Unknown error creating tensor from binary data");
  }
}

NIF(scalar_tensor) {
  SCALAR_PARAM(0, scalar);
  TYPE_PARAM(1, type);
  // DEVICE_PARAM(2, device);

  TENSOR(mlx::core::array(scalar, type))
}

NIF(eye) {
  PARAM(0, int, m);
  PARAM(1, int, n);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::eye(m, n, 0, type, device));
}

NIF(broadcast_to) {
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);
  DEVICE_PARAM(2, device);

  auto result = mlx::core::broadcast_to(*t, shape, device);

  TENSOR(result);
}

NIF(tensordot) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  LIST_PARAM(2, std::vector<int>, axes1);
  LIST_PARAM(3, std::vector<int>, axes2);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::tensordot(*a, *b, axes1, axes2, device));
}

/* Unary Ops */

#define UNARY_OP(OP) UNARY_OP2(OP, OP)

#define UNARY_OP2(OP, NATIVE_OP)                                               \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    DEVICE_PARAM(1, device);                                                   \
                                                                               \
    TENSOR(mlx::core::NATIVE_OP(*tensor, device));                             \
  }

/* Binary Ops */

#define BINARY_OP(OP) BINARY_OP2(OP, OP)

#define BINARY_OP2(OP, NATIVE_OP)                                              \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, a);                                                        \
    TENSOR_PARAM(1, b);                                                        \
    DEVICE_PARAM(2, device);                                                   \
                                                                               \
    TENSOR(mlx::core::NATIVE_OP(*a, *b, device));                              \
  }

static void free_tensor(ErlNifEnv *env, void *obj) {
  mlx::core::array *arr = static_cast<mlx::core::array *>(obj);
  if (arr != nullptr) {
    arr->~array();
  }
}

static int open_resource_type(ErlNifEnv *env) {
  const char *name = "MLXArray";
  ErlNifResourceFlags flags =
      (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

  TENSOR_TYPE =
      enif_open_resource_type(env, NULL, name, free_tensor, flags, NULL);
  if (TENSOR_TYPE == NULL) {
    return -1;
  }
  return 0;
}

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  if (open_resource_type(env) != 0) {
    return -1;
  }
  return 0;
}

UNARY_OP(abs)
UNARY_OP(ceil)
UNARY_OP(conjugate)
UNARY_OP(floor)
UNARY_OP2(negate, negative)
UNARY_OP(round)
UNARY_OP(sign)
UNARY_OP(real)
UNARY_OP(imag)
UNARY_OP2(is_nan, isnan)
UNARY_OP2(is_infinity, isinf)
UNARY_OP(logical_not)
UNARY_OP(sigmoid)
BINARY_OP(add)
BINARY_OP(subtract)
BINARY_OP(multiply)
BINARY_OP2(pow, power)
BINARY_OP2(remainder, remainder)
BINARY_OP2(divide, divide)
BINARY_OP2(atan2, arctan2)
BINARY_OP2(min, minimum)
BINARY_OP2(max, maximum)
BINARY_OP(bitwise_and)
BINARY_OP(bitwise_or)
BINARY_OP(bitwise_xor)
BINARY_OP(left_shift)
BINARY_OP(right_shift)
BINARY_OP(equal)
BINARY_OP(not_equal)
BINARY_OP(greater)
BINARY_OP(less)
BINARY_OP(greater_equal)
BINARY_OP(less_equal)
BINARY_OP(logical_and)
BINARY_OP(logical_or)

static ErlNifFunc nif_funcs[] = {{"scalar_type", 1, scalar_type},
                                 {"all", 4, all},
                                 {"any", 4, any},
                                 {"sum", 4, sum},
                                 {"product", 4, product},
                                 {"shape", 1, shape},
                                 {"reshape", 3, reshape},
                                 {"to_type", 3, to_type},
                                 {"to_blob", 1, to_blob},
                                 {"to_blob", 2, to_blob},
                                 {"from_blob", 4, from_blob},
                                 {"scalar_tensor", 3, scalar_tensor},
                                 {"ones", 3, ones},
                                 {"eye", 4, eye},
                                 {"broadcast_to", 3, broadcast_to},
                                 {"tensordot", 5, tensordot},
                                 {"abs", 2, abs},
                                 {"ceil", 2, ceil},
                                 {"conjugate", 2, conjugate},
                                 {"floor", 2, floor},
                                 {"negate", 2, negate},
                                 {"round", 2, round},
                                 {"sign", 2, sign},
                                 {"real", 2, real},
                                 {"imag", 2, imag},
                                 {"is_nan", 2, is_nan},
                                 {"is_infinity", 2, is_infinity},
                                 {"logical_not", 2, logical_not},
                                 {"sigmoid", 2, sigmoid},
                                 {"add", 3, add},
                                 {"subtract", 3, subtract},
                                 {"multiply", 3, multiply},
                                 {"pow", 3, pow},
                                 {"remainder", 3, remainder},
                                 {"divide", 3, divide},
                                 {"atan2", 3, atan2},
                                 {"bitwise_and", 3, bitwise_and},
                                 {"bitwise_or", 3, bitwise_or},
                                 {"bitwise_xor", 3, bitwise_xor},
                                 {"left_shift", 3, left_shift},
                                 {"right_shift", 3, right_shift},
                                 {"min", 3, min},
                                 {"max", 3, max},
                                 {"equal", 3, equal},
                                 {"not_equal", 3, not_equal},
                                 {"greater", 3, greater},
                                 {"less", 3, less},
                                 {"greater_equal", 3, greater_equal},
                                 {"less_equal", 3, less_equal},
                                 {"logical_and", 3, logical_and},
                                 {"logical_or", 3, logical_or},
                                 {"deallocate", 1, deallocate}};

// Update the NIF initialization
ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, NULL, NULL)
