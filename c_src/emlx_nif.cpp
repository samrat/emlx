#include "erl_nif.h"
#include "mlx/backend/common/utils.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <string>

#define NIF_CALL_IMPLEMENTATION
#include "nif_call.h"

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
    if (!enif_get_resource(env, arg, resource_object<mlx::core::array>::type,
                           (void **)&ptr)) {
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
      resource_object<mlx::core::array>::type, sizeof(mlx::core::array) +
                                                   sizeof(std::atomic<int>) +
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

ERL_NIF_TERM create_function_resource(ErlNifEnv *env, emlx::function function) {
  ERL_NIF_TERM ret;
  std::atomic<int> *refcount;
  auto function_ptr = (emlx::function *)enif_alloc_resource(
      resource_object<emlx::function>::type,
      sizeof(std::function<std::vector<array>(const std::vector<array> &)>) +
          sizeof(std::atomic<int>) + sizeof(std::atomic_flag));

  if (function_ptr == NULL) {
    return enif_make_badarg(env);
  }

  new (function_ptr) emlx::function(function);
  refcount = new (function_ptr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, function_ptr);
  enif_release_resource(function_ptr);

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

NIF(astype) {
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
  SCALAR_PARAM(0, scalar, is_complex);
  TYPE_PARAM(1, type);
  // DEVICE_PARAM(2, device);

  if (is_complex) {
    TENSOR(mlx::core::array(complex_scalar, type))
  } else {
    TENSOR(mlx::core::array(scalar, type))
  }
}

NIF(full) {
  SCALAR_PARAM(0, scalar, is_complex);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  if (is_complex) {
    TENSOR(mlx::core::full(shape, complex_scalar, type, device));
  } else {
    TENSOR(mlx::core::full(shape, scalar, type, device));
  }
}

NIF(arange) {
  PARAM(0, int, start);
  PARAM(1, int, stop);
  PARAM(2, int, step);
  PARAM(3, bool, integer);
  DEVICE_PARAM(4, device);

  if (integer) {
    TENSOR(mlx::core::arange(start, stop, step, device));
  } else {
    TENSOR(mlx::core::arange(static_cast<double>(start),
                             static_cast<double>(stop),
                             static_cast<double>(step), device));
  }
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

NIF(einsum) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);

  std::string spec_string;
  if (!nx::nif::get(env, argv[2], spec_string)) {
    return nx::nif::error(env, "Unable to get spec_string param.");
  }

  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::einsum(spec_string, std::vector<mlx::core::array>({*a, *b}),
                           device));
}

NIF(tri_inv) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, bool, upper);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::linalg::tri_inv(*tensor, upper, device));
}

NIF(conv_general) {
  TENSOR_PARAM(0, tensor_input);
  TENSOR_PARAM(1, tensor_kernel);
  LIST_PARAM(2, std::vector<int>, strides);
  LIST_PARAM(3, std::vector<int>, padding_low);
  LIST_PARAM(4, std::vector<int>, padding_high);
  LIST_PARAM(5, std::vector<int>, kernel_dilation);
  LIST_PARAM(6, std::vector<int>, input_dilation);
  PARAM(7, int, feature_group_count);
  DEVICE_PARAM(8, device);

  TENSOR(mlx::core::conv_general(
      *tensor_input, *tensor_kernel, strides, padding_low, padding_high,
      kernel_dilation, input_dilation, feature_group_count, false, device));
}

NIF(transpose) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::transpose(*t, axes, device));
}

NIF(pad) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  LIST_PARAM(2, std::vector<int>, low_pad_size);
  LIST_PARAM(3, std::vector<int>, high_pad_size);
  TENSOR_PARAM(4, pad_value);
  DEVICE_PARAM(5, device);

  TENSOR(mlx::core::pad(*t, axes, low_pad_size, high_pad_size, *pad_value,
                        "constant", device))
};

NIF(sort) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::sort(*t, axis, device));
}

NIF(argsort) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::argsort(*t, axis, device));
}

NIF(eval) {
  TENSOR_PARAM(0, t);
  mlx::core::eval(*t);
  return nx::nif::ok(env);
}

NIF(set_compile) {
  PARAM(0, bool, compile);

  if (compile) {
    mlx::core::enable_compile();
  } else {
    mlx::core::disable_compile();
  }

  return nx::nif::ok(env);
}

void move_between_envs(ERL_NIF_TERM from_term, ERL_NIF_TERM *to_term,
                       ErlNifEnv *from_env, ErlNifEnv *to_env) {
  ErlNifBinary serialized;
  enif_term_to_binary(from_env, from_term, &serialized);
  enif_binary_to_term(to_env, serialized.data, serialized.size, to_term, 0);
}

NIF(compile) {
  ERL_NIF_TERM callback_fun_outer = argv[0];
  LIST_PARAM(1, std::vector<mlx::core::array>, arrays);
  ErlNifPid evaluator_pid;
  if (!enif_get_local_pid(env, argv[2], &evaluator_pid)) {
    return nx::nif::error(env, "Could not get evaluator pid");
  }

  ErlNifEnv *closure_env = enif_alloc_env();

  auto fun = [env = closure_env, evaluator_pid, callback_fun_outer](
                 const std::vector<mlx::core::array> &compile_args) {
    ERL_NIF_TERM callback_fun = enif_make_copy(env, callback_fun_outer);
    ERL_NIF_TERM tensor_list = nx::nif::make_list(env, compile_args);
    ERL_NIF_TERM arg_list = enif_make_list1(env, tensor_list);

    ERL_NIF_TERM output_list =
        make_nif_call(env, evaluator_pid, callback_fun, arg_list);

    // Convert output_list back to vector of MLX arrays
    std::vector<mlx::core::array> output_tensors;
    if (!nx::nif::get_list(env, output_list, output_tensors)) {
      throw std::runtime_error("Failed to convert callback result to tensors");
    }

    enif_free_env(env);

    return output_tensors;
  };

  emlx::function compiled_function_ptr = mlx::core::compile(fun);

  return nx::nif::ok(env, create_function_resource(env, compiled_function_ptr));
}

NIF(call_compiled) {
  emlx::function *compiled_function_ptr = nullptr;

  if (!nx::nif::get(env, argv[0], compiled_function_ptr)) {
    return nx::nif::error(env, "Unable to get compiled function pointer");
  }
  LIST_PARAM(1, std::vector<mlx::core::array>, args);

  std::vector<mlx::core::array> result = (*compiled_function_ptr)(args);

  return nx::nif::ok(env, nx::nif::make_list(env, result));
}

NIF(stack) {
  LIST_PARAM(0, std::vector<mlx::core::array>, arrays);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::stack(arrays, axis, device));
}

NIF(where) {
  TENSOR_PARAM(0, condition);
  TENSOR_PARAM(1, x);
  TENSOR_PARAM(2, y);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::where(*condition, *x, *y, device));
}

NIF(concatenate) {
  LIST_PARAM(0, std::vector<mlx::core::array>, arrays);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::concatenate(arrays, axis, device));
}

NIF(take_along_axis) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, indices);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::take_along_axis(*t, *indices, axis, device));
}

NIF(take) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, indices);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::take(*t, *indices, axis, device));
}

NIF(gather) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<mlx::core::array>, indices);
  LIST_PARAM(2, std::vector<int>, axes);
  LIST_PARAM(3, std::vector<int>, slice_sizes);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::gather(*t, indices, axes, slice_sizes, device));
}

NIF(scatter_add) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<mlx::core::array>, indices);
  TENSOR_PARAM(2, tensor_updates);
  LIST_PARAM(3, std::vector<int>, axes);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::scatter_add(*t, indices, *tensor_updates, axes, device));
}

NIF(scatter) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<mlx::core::array>, indices);
  TENSOR_PARAM(2, tensor_updates);
  LIST_PARAM(3, std::vector<int>, axes);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::scatter(*t, indices, *tensor_updates, axes, device));
}

/* Reduction Ops */

#define REDUCTION_AXES_OP(OP) REDUCTION_AXES_OP2(OP, OP)

#define REDUCTION_AXES_OP2(OP, NATIVE_OP)                                      \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    LIST_PARAM(1, std::vector<int>, axes);                                     \
    PARAM(2, bool, keep_dims);                                                 \
    DEVICE_PARAM(3, device);                                                   \
                                                                               \
    if (axes.empty()) {                                                        \
      for (int i = 0; i < tensor->ndim(); ++i) {                               \
        axes.push_back(i);                                                     \
      }                                                                        \
    }                                                                          \
    TENSOR(mlx::core::NATIVE_OP(*tensor, axes, keep_dims, device));            \
  }

#define REDUCTION_AXIS_OP(OP) REDUCTION_AXIS_OP2(OP, OP)

#define REDUCTION_AXIS_OP2(OP, NATIVE_OP)                                      \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    if (argc == 3) {                                                           \
      PARAM(1, bool, keep_dims);                                               \
      DEVICE_PARAM(2, device);                                                 \
      TENSOR(mlx::core::NATIVE_OP(*tensor, keep_dims, device));                \
    } else {                                                                   \
      PARAM(1, int, axis);                                                     \
      PARAM(2, bool, keep_dims);                                               \
      DEVICE_PARAM(3, device);                                                 \
      TENSOR(mlx::core::NATIVE_OP(*tensor, axis, keep_dims, device));          \
    }                                                                          \
  }

#define REDUCTION_AXIS_REVERSIBLE_OP(OP) REDUCTION_AXIS_REVERSIBLE_OP2(OP, OP)

#define REDUCTION_AXIS_REVERSIBLE_OP2(OP, NATIVE_OP)                           \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    PARAM(1, int, axis);                                                       \
    PARAM(2, bool, keep_dims);                                                 \
    DEVICE_PARAM(3, device);                                                   \
                                                                               \
    TENSOR(mlx::core::NATIVE_OP(*tensor, axis, keep_dims, device));            \
  }

REDUCTION_AXES_OP(all)
REDUCTION_AXES_OP(any)
REDUCTION_AXES_OP(sum)
REDUCTION_AXES_OP2(product, prod)
REDUCTION_AXIS_OP(argmax)
REDUCTION_AXIS_OP(argmin)

NIF(cumulative_sum) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cumsum(*tensor, axis, reverse, inclusive, device));
}

NIF(cumulative_product) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cumprod(*tensor, axis, reverse, inclusive, device));
}

NIF(cumulative_max) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cummax(*tensor, axis, reverse, inclusive, device));
}

NIF(cumulative_min) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cummin(*tensor, axis, reverse, inclusive, device));
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

static int open_resources(ErlNifEnv *env) {
  const char *mod = "EMLX";
  if (!open_resource<mlx::core::array>(env, mod, "MLXArray")) {
    return -1;
  }

  if (!open_resource<emlx::function>(env, mod, "CompiledFunction")) {
    return -1;
  }

  return 0;
}

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  if (open_resources(env) != 0) {
    return -1;
  }

  if (nif_call_onload(env) != 0) {
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

UNARY_OP2(asin, arcsin)
UNARY_OP2(asinh, arcsinh)
UNARY_OP2(acos, arccos)
UNARY_OP2(acosh, arccosh)
UNARY_OP2(atan, arctan)
UNARY_OP2(atanh, arctanh)
UNARY_OP(cos)
UNARY_OP(cosh)
UNARY_OP(erf)
UNARY_OP2(erf_inv, erfinv)
UNARY_OP(exp)
UNARY_OP(expm1)
UNARY_OP(log)
UNARY_OP(log1p)
UNARY_OP(rsqrt)
UNARY_OP(sin)
UNARY_OP(sinh)
UNARY_OP(sqrt)
UNARY_OP(tan)
UNARY_OP(tanh)

BINARY_OP(add)
BINARY_OP(subtract)
BINARY_OP(multiply)
BINARY_OP2(pow, power)
BINARY_OP2(remainder, remainder)
BINARY_OP2(divide, divide)
BINARY_OP2(atan2, arctan2)
BINARY_OP2(minimum, minimum)
BINARY_OP2(maximum, maximum)
BINARY_OP2(quotient, floor_divide)
BINARY_OP(bitwise_and)
BINARY_OP(bitwise_or)
BINARY_OP(bitwise_xor)
NIF(bitwise_not) {
  TENSOR_PARAM(0, a);
  DEVICE_PARAM(1, device);

  auto dtype = (*a).dtype();
  auto mask = mlx::core::full({}, 0xFFFFFFFFFFFFFFFF, dtype, device);
  TENSOR(mlx::core::subtract(mask, *a, device));
}
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
NIF(logical_xor) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  DEVICE_PARAM(2, device);

  auto t1 = mlx::core::logical_or(*a, *b, device);
  auto t2 =
      mlx::core::logical_not(mlx::core::logical_and(*a, *b, device), device);
  TENSOR(mlx::core::logical_and(t1, t2, device));
}
NIF(allclose) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  PARAM(2, double, rtol);
  PARAM(3, double, atol);
  PARAM(4, bool, equal_nan);
  DEVICE_PARAM(5, device);

  TENSOR(mlx::core::allclose(*a, *b, rtol, atol, equal_nan, device));
}
NIF(isclose) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  PARAM(2, double, rtol);
  PARAM(3, double, atol);
  PARAM(4, bool, equal_nan);
  DEVICE_PARAM(5, device);

  TENSOR(mlx::core::isclose(*a, *b, rtol, atol, equal_nan, device));
}

NIF(item) {
  TENSOR_PARAM(0, t);
  mlx::core::eval(*t);
  auto dtype_kind = mlx::core::kindof(t->dtype());

  if (dtype_kind == mlx::core::Dtype::Kind::u ||
      dtype_kind == mlx::core::Dtype::Kind::i ||
      dtype_kind == mlx::core::Dtype::Kind::b) {
    int64_t value = t->item<int64_t>();
    return nx::nif::ok(env, nx::nif::make(env, value));
  } else {
    double value = t->item<double>();
    return nx::nif::ok(env, nx::nif::make(env, value));
  }
}

NIF(slice) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, starts);
  LIST_PARAM(2, std::vector<int>, stops);
  LIST_PARAM(3, std::vector<int>, strides);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::slice(*t, starts, stops, strides, device));
}

NIF(slice_update) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, tensor_updates);
  LIST_PARAM(2, std::vector<int>, starts);
  LIST_PARAM(3, std::vector<int>, stops);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::slice_update(*t, *tensor_updates, starts, stops, device));
}

NIF(squeeze) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  DEVICE_PARAM(2, device);
  TENSOR(mlx::core::squeeze(*t, axes, device));
}

NIF(emlx_fft) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, n);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::fft(*t, n, axis, device));
}

NIF(ifft) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, n);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::ifft(*t, n, axis, device));
}

NIF(emlx_fft2) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, n);
  LIST_PARAM(2, std::vector<int>, axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::fft2(*t, n, axes, device));
}

NIF(ifft2) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, n);
  LIST_PARAM(2, std::vector<int>, axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::ifft2(*t, n, axes, device));
}

NIF(view) {
  TENSOR_PARAM(0, t);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);
  TENSOR(mlx::core::view(*t, type, device));
}

NIF(max) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::max(*t, axes, keep_axes, device));
}

NIF(min) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::min(*t, axes, keep_axes, device));
}

NIF(clip) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, min);
  TENSOR_PARAM(2, max);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::clip(*t, *min, *max, device));
}

NIF(strides) {
  TENSOR_PARAM(0, t);

  auto strides = t->strides();

  return nx::nif::ok(env, nx::nif::make_list(env, strides));
}

NIF(as_strided) {
  TENSOR_PARAM(0, t);
  TUPLE_PARAM(1, std::vector<int>, shape);
  LIST_PARAM(2, std::vector<size_t>, strides);
  PARAM(3, int, offset);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::as_strided(*t, shape, strides, offset, device));
}

static ErlNifFunc nif_funcs[] = {
    NIF_CALL_NIF_FUNC(nif_call_evaluated),
    {"strides", 1, strides},
    {"as_strided", 5, as_strided},
    {"scalar_type", 1, scalar_type},
    {"eval", 1, eval},
    {"view", 3, view},
    {"stack", 3, stack},
    {"where", 4, where},
    {"concatenate", 3, concatenate},
    {"take_along_axis", 4, take_along_axis},
    {"take", 4, take},
    {"gather", 5, gather},
    {"scatter_add", 5, scatter_add},
    {"scatter", 5, scatter},
    {"slice", 5, slice},
    {"slice_update", 5, slice_update},
    {"squeeze", 3, squeeze},
    {"item", 1, item},
    {"all", 4, all},
    {"any", 4, any},
    {"sum", 4, sum},
    {"product", 4, product},
    {"argmax", 3, argmax},
    {"argmax", 4, argmax},
    {"argmin", 3, argmin},
    {"argmin", 4, argmin},
    {"cumulative_sum", 5, cumulative_sum},
    {"cumulative_product", 5, cumulative_product},
    {"cumulative_max", 5, cumulative_max},
    {"cumulative_min", 5, cumulative_min},
    {"shape", 1, shape},
    {"reshape", 3, reshape},
    {"astype", 3, astype},
    {"to_blob", 1, to_blob},
    {"to_blob", 2, to_blob},
    {"from_blob", 4, from_blob},
    {"scalar_tensor", 3, scalar_tensor},
    {"ones", 3, ones},
    {"full", 4, full},
    {"arange", 5, arange},
    {"eye", 4, eye},
    {"broadcast_to", 3, broadcast_to},
    {"tensordot", 5, tensordot},
    {"einsum", 4, einsum},
    {"conv_general", 9, conv_general},
    {"transpose", 3, transpose},
    {"pad", 6, pad},
    {"sort", 3, sort},
    {"argsort", 3, argsort},
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
    {"asin", 2, asin},
    {"asinh", 2, asinh},
    {"acos", 2, acos},
    {"acosh", 2, acosh},
    {"cos", 2, cos},
    {"cosh", 2, cosh},
    {"atan", 2, atan},
    {"atanh", 2, atanh},
    {"erf", 2, erf},
    {"erf_inv", 2, erf_inv},
    {"exp", 2, exp},
    {"expm1", 2, expm1},
    {"log", 2, log},
    {"log1p", 2, log1p},
    {"rsqrt", 2, rsqrt},
    {"sin", 2, sin},
    {"sinh", 2, sinh},
    {"sqrt", 2, sqrt},
    {"tan", 2, tan},
    {"tanh", 2, tanh},
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
    {"bitwise_not", 2, bitwise_not},
    {"left_shift", 3, left_shift},
    {"right_shift", 3, right_shift},
    {"minimum", 3, minimum},
    {"maximum", 3, maximum},
    {"quotient", 3, quotient},
    {"equal", 3, equal},
    {"not_equal", 3, not_equal},
    {"greater", 3, greater},
    {"less", 3, less},
    {"greater_equal", 3, greater_equal},
    {"less_equal", 3, less_equal},
    {"logical_and", 3, logical_and},
    {"logical_or", 3, logical_or},
    {"logical_xor", 3, logical_xor},
    {"fft", 4, emlx_fft},
    {"ifft", 4, ifft},
    {"fft2", 4, emlx_fft2},
    {"ifft2", 4, ifft2},
    {"allclose", 6, allclose},
    {"isclose", 6, isclose},
    {"deallocate", 1, deallocate},
    {"max", 4, max},
    {"min", 4, min},
    {"clip", 4, clip},
    {"tri_inv", 3, tri_inv},
    {"set_compile", 1, set_compile},
    {"compile", 3, compile, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"call_compiled", 2, call_compiled, ERL_NIF_DIRTY_JOB_CPU_BOUND}};

// Update the NIF initialization
ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, NULL, NULL)
