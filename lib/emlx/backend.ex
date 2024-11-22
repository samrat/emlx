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
    |> maybe_pad_binary(type)
    |> EMLX.from_blob(
      shape,
      to_mlx_type(type),
      device_option(backend_options)
    )
    |> to_nx(out)
  end

  defp maybe_pad_binary(bin, {:u, size}) when size in [16, 32] do
    double_size = size * 2
    for <<x::native-size(size) <- bin>>, into: <<>>, do: <<x::native-size(double_size)>>
  end

  defp maybe_pad_binary(bin, {:u, size}) when size in [2, 4] do
    for <<x::native-size(size) <- bin>>, into: <<>>, do: <<x::native-8>>
  end

  defp maybe_pad_binary(bin, {:s, size}) when size in [2, 4] do
    for <<x::native-signed-size(size) <- bin>>, into: <<>>, do: <<x::native-signed-8>>
  end

  defp maybe_pad_binary(bin, _), do: bin

  defp maybe_add_signature(result, %T{data: %Backend{ref: _}}) do
    Inspect.Algebra.concat([
      "EMLX.Backend",
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
  defp to_mlx_type({:bf, 16}), do: :float32
  defp to_mlx_type(:bool), do: :bfloat16

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
  defp to_nx_type(:bool), do: :bool

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

  # FIXME: Use `full` like torchx
  # def constant(%T{shape: shape, type: type} = out, scalar, _backend_options) do
  #   shape_list = Tuple.to_list(shape)

  #   scalar
  #   |> constant_serialize_scalar()
  # end

  @impl true
  def sum(%T{} = out, %T{} = t, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    # Calculate the expected output shape based on the input shape and axes
    result =
      t
      |> from_nx()
      |> EMLX.sum(axes, keep_axes)
      |> EMLX.to_type(to_mlx_type(out.type))

    # Get the actual shape after summation
    actual_shape = EMLX.shape(result)
    # FIXME: MLX returns whatever the original type is, but Nx expects u8 -> u32
    scalar_type = EMLX.scalar_type(result)

    # Create a new output tensor with the correct shape
    %{out | shape: actual_shape}
    |> then(&to_nx(result, &1))
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
        _left_batched_axes,
        %T{type: right_type} = right,
        right_axes,
        _right_batched_axes
      ) do
    left_tx = from_nx(left)
    right_tx = from_nx(right)

    EMLX.tensordot(
      to_typed_ref(left_tx, left_type, out_type),
      to_typed_ref(right_tx, right_type, out_type),
      left_axes,
      right_axes
    )
    |> to_nx(out)
  end

  # Unary Ops

  ops = [:abs, :ceil, :conjugate, :floor, :negate, :round, :sign, :real, :imag, :logical_not]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor) do
      EMLX.unquote(op)(from_nx(tensor)) |> to_nx(out)
    end
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

  defp bitmask({device, _} = tensor, {:u, 16}),
    do: EMLX.bitwise_and(tensor, EMLX.scalar_tensor(0xFFFF, :int, device))

  defp bitmask({device, _} = tensor, {:u, 32}),
    do: EMLX.bitwise_and(tensor, EMLX.scalar_tensor(0xFFFF_FFFF, :int64, device))

  defp bitmask(tensor, {_, _}),
    do: tensor

  ops =
    [:min, :max, :divide, :quotient, :remainder, :atan2] ++
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

  # Helper function to handle different scalar types
  defp constant_serialize_scalar(scalar) when is_number(scalar), do: scalar
  defp constant_serialize_scalar(%Complex{} = c), do: Complex.abs(c)

  defp to_typed_ref(tensor, expected_type, expected_type),
    do: tensor

  defp to_typed_ref(tensor, ref_type, expected_type),
    do: EMLX.to_type(tensor, to_mlx_type(expected_type))

  defp device_option(nil), do: :cpu
  defp device_option(backend_opts), do: backend_opts[:device] || :cpu
end
