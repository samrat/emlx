defmodule EMLX.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias EMLX.Backend, as: Backend

  defstruct [:ref, :shape, :type, :data]

  @impl true
  def init(opts) do
    Keyword.validate!(opts, device: :cpu)
  end

  @doc """
  Converts from an Nx tensor to an MLX array.
  """
  def from_nx(%T{data: %Backend{ref: device_ref}}) do
    device_ref
  end

  def from_nx(%T{} = other_backend), do: Nx.backend_transfer(other_backend, Backend) |> from_nx()

  @doc """
  Converts an MLX array back to an Nx tensor.
  """
  def to_nx({device, ref} = device_ref, %T{type: type, shape: shape} = t)
      when is_atom(device) and is_reference(ref) do
    # Get the MLX array's type
    mlx_type = EMLX.scalar_type(device_ref)

    # Convert if needed (similar to the torch byte conversion)
    array =
      if needs_type_conversion?(type, mlx_type) do
        EMLX.to_type(device_ref, to_mlx_type(type))
      else
        device_ref
      end

    %T{
      t
      | data: %Backend{ref: check_shape_and_type!(array, shape, type), shape: shape, type: type}
    }
  end

  @impl true
  def backend_copy(%T{type: type, shape: shape} = tensor, backend, opts) do
    Nx.from_binary(to_binary(tensor, Nx.size(tensor)), type, backend: {backend, opts})
    |> Nx.reshape(shape)
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    new_tensor = backend_copy(tensor, backend, opts)
    backend_deallocate(tensor)
    new_tensor
  end

  @impl true
  def backend_deallocate(%T{data: %Backend{ref: ref}}) do
    EMLX.deallocate(ref)
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  @impl true
  def to_binary(tensor, limit) do
    EMLX.to_blob(from_nx(tensor), limit)
  end

  @impl true
  def from_binary(%T{type: type, shape: shape} = out, binary, backend_options) do
    binary
    |> EMLX.from_blob(
      shape,
      to_mlx_type(type),
      device_option(backend_options)
    )
    |> to_nx(out)
  end

  @impl true
  def slice(
        out,
        %T{shape: input_shape} = t,
        start_indices,
        lengths,
        strides
      ) do
    t
    |> from_nx()
    |> mlx_slice(input_shape, start_indices, lengths, strides)
    |> to_nx(out)
  end

  defp mlx_slice(t, input_shape, start_indices, lengths, strides) do
    starts =
      start_indices
      |> Enum.zip(lengths)
      |> Enum.with_index(fn {start, len}, axis ->
        min(to_number(start), elem(input_shape, axis) - len)
      end)

    stops = Enum.zip_with(starts, lengths, &(&1 + &2))

    EMLX.slice(t, starts, stops, strides)
  end

  @impl true
  def squeeze(out, tensor, axes) do
    tensor
    |> from_nx()
    |> EMLX.squeeze(axes)
    |> to_nx(out)
  end

  @impl true
  def transpose(out, tensor, axes) do
    tensor
    |> from_nx()
    |> EMLX.transpose(axes)
    |> to_nx(out)
  end

  @impl true
  def bitcast(out, tensor) do
    tensor
    |> from_nx()
    |> EMLX.view(to_mlx_type(out.type))
    |> to_nx(out)
  end

  defp maybe_add_signature(result, %T{data: %Backend{ref: {device, ref}}}) do
    ~c"#Ref<" ++ rest = :erlang.ref_to_list(ref)

    Inspect.Algebra.concat([
      "EMLX.Backend<#{device}, ",
      List.to_string(rest),
      Inspect.Algebra.line(),
      result
    ])
  end

  # Helper functions
  defp needs_type_conversion?({:u, 8}, :bool), do: true
  defp needs_type_conversion?(_, _), do: false

  defp to_mlx_type({:u, 8}), do: :uint8
  defp to_mlx_type({:u, 16}), do: :uint16
  defp to_mlx_type({:u, 32}), do: :uint32
  defp to_mlx_type({:u, 64}), do: :uint64
  defp to_mlx_type({:s, 8}), do: :int8
  defp to_mlx_type({:s, 16}), do: :int16
  defp to_mlx_type({:s, 32}), do: :int32
  defp to_mlx_type({:s, 64}), do: :int64
  defp to_mlx_type({:f, 16}), do: :float16
  defp to_mlx_type({:f, 32}), do: :float32
  defp to_mlx_type({:bf, 16}), do: :bfloat16
  defp to_mlx_type({:c, 64}), do: :complex64
  defp to_mlx_type(:bool), do: :bool

  defp to_nx_type(:uint8), do: {:u, 8}
  defp to_nx_type(:uint16), do: {:u, 16}
  defp to_nx_type(:uint32), do: {:u, 32}
  defp to_nx_type(:uint64), do: {:u, 64}
  defp to_nx_type(:int8), do: {:s, 8}
  defp to_nx_type(:int16), do: {:s, 16}
  defp to_nx_type(:int32), do: {:s, 32}
  defp to_nx_type(:int64), do: {:s, 64}
  defp to_nx_type(:float16), do: {:f, 16}
  defp to_nx_type(:float32), do: {:f, 32}
  defp to_nx_type(:bfloat16), do: {:bf, 16}
  defp to_nx_type(:complex64), do: {:c, 64}
  defp to_nx_type(:bool), do: :bool

  defp to_number(n) when is_number(n), do: n
  defp to_number(%T{} = t), do: t |> from_nx() |> EMLX.item()

  defp check_shape_and_type!(device_ref, expected_shape, expected_type) do
    actual_shape = EMLX.shape(device_ref)
    actual_type = EMLX.scalar_type(device_ref) |> to_nx_type()

    if actual_shape != expected_shape do
      raise ArgumentError, """
      Shape mismatch in MLX array conversion:
      Expected shape: #{inspect(expected_shape)}
      Got shape: #{inspect(actual_shape)}
      """
    end

    case {actual_type, expected_type} do
      {{:s, 8}, {:s, qint}} when qint in [2, 4] ->
        :ok

      {{:u, 8}, {:u, qint}} when qint in [2, 4] ->
        :ok

      {{:s, 32}, {:u, 16}} ->
        :ok

      {{:s, 64}, {:u, 32}} ->
        :ok

      {{:s, 64}, {:u, 64}} ->
        :ok

      {{:u, 8}, {:u, 32}} ->
        :ok

      _ when actual_type != expected_type ->
        raise "type mismatch in EMLX: expected #{inspect(expected_type)}, got: #{inspect(actual_type)}. " <>
                "Please report this bug"

      _ ->
        :ok
    end

    device_ref
  end

  @impl true
  def constant(
        %T{shape: shape, names: names, type: type} = out,
        scalar,
        backend_options
      )
      when scalar in [:infinity, :neg_infinity, :nan] do
    t = apply(Nx.Constants, scalar, [type, [backend: {Backend, backend_options}]])
    Nx.broadcast(t, shape, names: names)
  end

  @impl true
  def constant(%T{shape: {}, type: type} = out, scalar, backend_options) do
    scalar
    |> constant_serialize_scalar()
    |> EMLX.scalar_tensor(to_mlx_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  def constant(%T{shape: shape, type: type} = out, scalar, backend_options) do
    scalar
    |> constant_serialize_scalar()
    |> EMLX.full(shape, to_mlx_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  @impl true
  def iota(%{shape: {}, type: type} = out, nil, backend_options) do
    constant(out, 0, backend_options)
  end

  def iota(%T{shape: shape, type: type} = out, nil, backend_options) do
    EMLX.arange(
      0,
      Nx.size(shape),
      1,
      Nx.Type.integer?(type),
      device_option(backend_options)
    )
    |> EMLX.to_type(to_mlx_type(type))
    |> EMLX.reshape(shape)
    |> to_nx(out)
  end

  @impl true
  def iota(%T{shape: {n}, type: type} = out, 0, backend_options) do
    EMLX.arange(0, n, 1, Nx.Type.integer?(type), device_option(backend_options))
    |> EMLX.to_type(to_mlx_type(type))
    |> to_nx(out)
  end

  def iota(%T{shape: shape, type: type} = out, axis, backend_options) do
    # gets the size of iota
    dim = elem(shape, axis)

    # build the iota in one dimension
    aten =
      EMLX.arange(0, dim, 1, Nx.Type.integer?(type), device_option(backend_options))
      |> EMLX.to_type(to_mlx_type(type))

    # reshape the tensor above to be have shape where everything is 1, except for dim
    reshape = Tuple.duplicate(1, Nx.rank(shape)) |> put_elem(axis, dim)
    aten = EMLX.reshape(aten, reshape)

    # Now broadcast the tensor using the original shape
    EMLX.broadcast_to(aten, shape) |> to_nx(out)
  end

  # Aggregation (axes)
  ops = [:all, :any, :sum, :product]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axes = opts[:axes] || []
      keep_axes = opts[:keep_axes] || false

      # Calculate the expected output shape based on the input shape and axes
      result =
        tensor
        |> from_nx()
        |> EMLX.unquote(op)(axes, keep_axes)
        |> EMLX.to_type(to_mlx_type(out.type))

      # Get the actual shape after summation
      actual_shape = EMLX.shape(result)
      # FIXME: MLX returns whatever the original type is, but Nx expects u8 -> u32
      # scalar_type = EMLX.scalar_type(result)

      # Create a new output tensor with the correct shape
      %{out | shape: actual_shape}
      |> then(&to_nx(result, &1))
    end
  end

  # Aggregation (axis)
  ops = [:argmax, :argmin]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axis = opts[:axis] || 0
      keep_axes = opts[:keep_axes] || false

      # Calculate the expected output shape based on the input shape and axes
      result =
        tensor
        |> from_nx()
        |> EMLX.unquote(op)(axis, keep_axes)
        |> EMLX.to_type(to_mlx_type(out.type))

      # Get the actual shape after summation
      actual_shape = EMLX.shape(result)
      # FIXME: MLX returns whatever the original type is, but Nx expects u8 -> u32
      # scalar_type = EMLX.scalar_type(result)

      # Create a new output tensor with the correct shape
      %{out | shape: actual_shape}
      |> then(&to_nx(result, &1))
    end
  end

  ops = [:cumulative_sum, :cumulative_product, :cumulative_max, :cumulative_min]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axis = opts[:axis] || 0
      reverse = opts[:reverse] || false

      # Calculate the expected output shape based on the input shape and axes
      inclusive = true

      result =
        tensor
        |> from_nx()
        |> EMLX.unquote(op)(axis, reverse, inclusive)
        |> EMLX.to_type(to_mlx_type(out.type))

      # Get the actual shape after summation
      actual_shape = EMLX.shape(result)
      # FIXME: MLX returns whatever the original type is, but Nx expects u8 -> u32
      # scalar_type = EMLX.scalar_type(result)

      # Create a new output tensor with the correct shape
      %{out | shape: actual_shape}
      |> then(&to_nx(result, &1))
    end
  end

  @impl true
  def stack(out, tensors, axis) do
    tensors
    |> Enum.map(&from_nx/1)
    |> EMLX.stack(axis)
    |> to_nx(out)
  end

  @impl true
  def concatenate(out, tensors, axis) do
    tensors
    |> Enum.map(&from_nx/1)
    |> EMLX.concatenate(axis)
    |> to_nx(out)
  end

  @impl true
  def put_slice(out, input, start_indices_unbounded, slice) do
    input_tx = from_nx(input)

    slice_shape_list = Tuple.to_list(slice.shape)

    zip_indices_input = [Tuple.to_list(input.shape), start_indices_unbounded, slice_shape_list]

    {start_indices, stop_indices} =
      Enum.zip_with(zip_indices_input, fn [dim_size, idx, len] ->
        idx = Nx.to_number(idx)
        start = min(max(idx, 0), dim_size - len)
        {start, start + len}
      end)
      |> Enum.unzip()

    slice_tx = slice |> from_nx() |> EMLX.to_type(to_mlx_type(out.type))

    input_tx
    |> EMLX.to_type(to_mlx_type(out.type))
    |> EMLX.slice_update(slice_tx, start_indices, stop_indices)
    |> to_nx(out)
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    on_true = Nx.as_type(on_true, Nx.type(out))
    on_false = Nx.as_type(on_false, Nx.type(out))
    on_true_torch = from_nx(on_true)
    on_false_torch = from_nx(on_false)

    # Use logical_not to convert any tensor to a boolean tensor
    # because of that, we have to swap true/false tensor
    pred
    |> from_nx()
    |> EMLX.logical_not()
    |> EMLX.where(on_false_torch, on_true_torch)
    |> to_nx(out)
  end

  @impl true
  def take_along_axis(out, tensor, idx, opts) do
    axis = opts[:axis]

    tensor
    |> from_nx()
    |> EMLX.take_along_axis(from_nx(idx), axis)
    |> to_nx(out)
  end

  @impl true
  def take(out, tensor, indices, opts) do
    axis = opts[:axis]

    tensor
    |> from_nx()
    |> EMLX.take(from_nx(indices), axis)
    |> to_nx(out)
  end

  @impl true
  def eye(%T{shape: shape, type: type} = out, backend_options) do
    rank = tuple_size(shape)
    m = elem(shape, rank - 2)
    n = elem(shape, rank - 1)

    EMLX.eye(m, n, to_mlx_type(type), device_option(backend_options))
    |> EMLX.broadcast_to(shape)
    |> to_nx(out)
  end

  @impl true
  def dot(
        %T{type: out_type} = out,
        %T{type: left_type} = left,
        left_axes,
        # MLX doesn't support batched axes
        left_batched_axes,
        %T{type: right_type} = right,
        right_axes,
        right_batched_axes
      ) do
    left_tx = from_nx(left)
    right_tx = from_nx(right)

    # TODO: MLX doesn't support batched axes, so we can do an outer loop in Elixir instead

    if left_batched_axes != [] or right_batched_axes != [] do
      raise "MLX doesn't support batched axes in tensordot"
    end

    if not Nx.Type.float?(out_type) do
      raise "MLX only supports floating point output types in tensordot"
    end

    EMLX.tensordot(
      to_typed_ref(left_tx, left_type, out_type),
      to_typed_ref(right_tx, right_type, out_type),
      left_axes,
      right_axes
    )
    |> to_nx(out)
  end

  # Unary Ops

  ops =
    [
      :abs,
      :ceil,
      :conjugate,
      :floor,
      :negate,
      :round,
      :sign,
      :real,
      :imag,
      :is_nan,
      :is_infinity,
      :logical_not,
      :bitwise_not
    ] ++
      [
        :sigmoid,
        :asin,
        :asinh,
        :acos,
        :acosh,
        :atan,
        :atanh,
        :cos,
        :cosh,
        :erf,
        :erf_inv,
        :exp,
        :expm1,
        :log,
        :log1p,
        :rsqrt,
        :sin,
        :sinh,
        :sqrt,
        :tan,
        :tanh
      ]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor) do
      EMLX.unquote(op)(from_nx(tensor)) |> to_nx(out)
    end
  end

  @impl true
  def erfc(out, tensor) do
    t = from_nx(tensor)

    out_type = to_mlx_type(out.type)
    {dev, _} = erf = EMLX.erf(t) |> EMLX.to_type(out_type)

    EMLX.scalar_tensor(1, out_type, dev)
    |> EMLX.subtract(erf)
    |> to_nx(out)
  end

  # Binary Ops

  ops = [:add, :subtract, :multiply, :pow, :left_shift]

  for op <- ops do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)
      {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)
      result = EMLX.unquote(op)(left_tx, right_tx)

      result
      |> bitmask(out.type)
      |> EMLX.to_type(to_mlx_type(out.type))
      |> to_nx(out)
    end
  end

  # FFT Ops
  @impl true
  def fft(out, tensor, opts) do
    length = opts[:length]
    axis = opts[:axis] || -1

    tensor
    |> from_nx()
    |> EMLX.fft(length, axis)
    |> to_nx(out)
  end

  @impl true
  def ifft(out, tensor, opts) do
    length = opts[:length]
    axis = opts[:axis] || -1

    tensor
    |> from_nx()
    |> EMLX.ifft(length, axis)
    |> to_nx(out)
  end

  @impl true
  def fft2(out, tensor, opts) do
    length = opts[:length]
    axes = opts[:axes] || [-2, -1]

    tensor
    |> from_nx()
    |> EMLX.fft2(length, axes)
    |> to_nx(out)
  end

  @impl true
  def ifft2(out, tensor, opts) do
    length = opts[:length]
    axes = opts[:axes] || [-2, -1]

    tensor
    |> from_nx()
    |> EMLX.ifft2(length, axes)
    |> to_nx(out)
  end

  @impl true
  def all_close(out, a, b, opts) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-8
    equal_nan = true

    EMLX.allclose(from_nx(a), from_nx(b), atol, rtol, equal_nan)
    |> to_nx(out)
  end

  defp bitmask({device, _} = tensor, {:u, 16}),
    do: EMLX.bitwise_and(tensor, EMLX.scalar_tensor(0xFFFF, :int, device))

  defp bitmask({device, _} = tensor, {:u, 32}),
    do: EMLX.bitwise_and(tensor, EMLX.scalar_tensor(0xFFFF_FFFF, :int64, device))

  defp bitmask(tensor, {_, _}),
    do: tensor

  ops =
    [:divide, :quotient, :remainder, :atan2] ++
      [:right_shift, :logical_and, :logical_or, :logical_xor] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor]

  for op <- ops do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)
      {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)

      EMLX.unquote(op)(left_tx, right_tx)
      |> EMLX.to_type(to_mlx_type(out.type))
      |> to_nx(out)
    end
  end

  @impl true
  def min(out, l, r) do
    {left, right} = maybe_upcast(l, r)
      {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)

      EMLX.minimum(left_tx, right_tx)
      |> EMLX.to_type(to_mlx_type(out.type))
      |> to_nx(out)
  end

  @impl true
  def max(out, l, r) do
    {left, right} = maybe_upcast(l, r)
      {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)

      EMLX.maximum(left_tx, right_tx)
      |> EMLX.to_type(to_mlx_type(out.type))
      |> to_nx(out)
  end

  defp maybe_upcast(%T{type: t} = left, %T{type: t} = right),
    do: {left, right}

  defp maybe_upcast(left, right) do
    type = Nx.Type.merge(left.type, right.type)
    {Nx.as_type(left, type), Nx.as_type(right, type)}
  end

  defp maybe_broadcast_bin_args(out_shape, l, r) do
    l_tx =
      case l.shape do
        ^out_shape ->
          from_nx(l)

        _ ->
          l |> from_nx() |> EMLX.broadcast_to(out_shape)
      end

    r_tx =
      case r.shape do
        ^out_shape -> from_nx(r)
        _ -> r |> from_nx() |> EMLX.broadcast_to(out_shape)
      end

    {l_tx, r_tx}
  end

  @impl true
  def reshape(%T{shape: shape} = out, %T{} = t),
    do: EMLX.reshape(from_nx(t), shape) |> to_nx(out)

  @impl true
  def broadcast(out, %T{} = t, shape, axes) do
    t
    |> maybe_reshape(shape, axes)
    |> from_nx()
    |> EMLX.broadcast_to(shape)
    |> to_nx(out)
  end

  defp maybe_reshape(%T{shape: {}} = t, target_shape, _axes) do
    shape = 1 |> List.duplicate(tuple_size(target_shape)) |> List.to_tuple()
    Nx.reshape(t, shape)
  end

  defp maybe_reshape(%T{shape: shape} = t, target_shape, axes) do
    base_broadcast_shape = 1 |> List.duplicate(tuple_size(target_shape)) |> List.to_tuple()

    new_shape =
      shape
      |> Tuple.to_list()
      |> Enum.zip(axes)
      |> Enum.reduce(base_broadcast_shape, fn {dim_size, target_axis}, shape_acc ->
        shape_acc
        |> Tuple.delete_at(target_axis)
        |> Tuple.insert_at(target_axis, dim_size)
      end)

    Nx.reshape(t, new_shape)
  end

  @impl true
  def as_type(%T{type: type} = out, %T{} = t),
    do: from_nx(t) |> EMLX.to_type(to_mlx_type(type)) |> bitmask(type) |> to_nx(out)

  @impl true
  def reduce_max(out, tensor, opts) do
    axes = opts[:axes] || Nx.axes(tensor)
    keep_axes = opts[:keep_axes]

    tensor
    |> from_nx()
    |> EMLX.max(axes, keep_axes)
    |> to_nx(out)
  end

  @impl true
  def reduce_min(out, tensor, opts) do
    axes = opts[:axes] || Nx.axes(tensor)
    keep_axes = opts[:keep_axes]

    tensor
    |> from_nx()
    |> EMLX.min(axes, keep_axes)
    |> to_nx(out)
  end

  # Helper function to handle different scalar types
  defp constant_serialize_scalar(scalar) when is_number(scalar), do: scalar
  defp constant_serialize_scalar(%Complex{} = c), do: Complex.abs(c)

  defp to_typed_ref(tensor, expected_type, expected_type),
    do: tensor

  defp to_typed_ref(tensor, ref_type, expected_type),
    do: EMLX.to_type(tensor, to_mlx_type(expected_type))

  defp device_option(nil), do: :cpu
  defp device_option(backend_opts), do: backend_opts[:device] || :cpu

  @impl true
  def reverse(out, tensor, axes) do
    shape = Tuple.to_list(tensor.shape)

    # For each dimension, set appropriate start/stop/stride values
    {starts, stops, strides} =
      shape
      |> Enum.with_index()
      |> Enum.map(fn {dim, idx} ->
        if idx in axes do
          # For reversed axes: start from end, stop at -1, stride backwards
          {dim - 1, -1, -1}
        else
          # For normal axes: start at 0, go to dim, stride forward
          {0, dim, 1}
        end
      end)
      |> Enum.unzip3()

    tensor
    |> from_nx()
    |> EMLX.slice(starts, stops, strides)
    |> to_nx(out)
  end
end
