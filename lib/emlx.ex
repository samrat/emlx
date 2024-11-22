defmodule EMLX.Macro do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__)
      Module.register_attribute(EMLX, :mlx_function, accumulate: true)
    end
  end

  @doc """
  Function that receives a device and allocates a tensor.
  """
  defmacro defdevice(call) do
    {name, args} = Macro.decompose_call(call)

    unless has_device?(args) do
      raise("At least one argument of defdevice function should be named 'device'.")
    end

    tensors =
      case tensors(args) do
        [] -> :ok
        tensors -> quote do: {unquote(tensors), _} = prepare_tensors!(unquote(tensors))
      end

    quote do
      @mlx_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        unquote(tensors)
        {user_device, index} = normalize_device!(var!(device))
        var!(device) = mlx_device!(user_device, index)

        EMLX.NIF.unquote(name)(unquote_splicing(args))
        |> unwrap_tensor!(user_device)
      end
    end
  end

  @doc """
  Generates a call that returns a tensor (or a tuple/list of tensors).

  All tensor variables must start with the name tensor.
  """
  defmacro deftensor(call) do
    defcall(call, :unwrap_tensor!, [Macro.var(:device, __MODULE__)])
  end

  @doc """
  Generates a call that returns a value (not a tensor).

  All tensor variables must start with the name tensor.
  """
  defmacro defvalue(call) do
    defcall(call, :unwrap!, [])
  end

  defp defcall(call, unwrapper, extra) do
    {name, args} = Macro.decompose_call(call)
    tensors = tensors(args)

    if tensors == [] do
      raise ArgumentError, "at least one tensor required in #{name}/#{length(args)}"
    end

    quote do
      @mlx_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        {unquote(tensors), device} = prepare_tensors!(unquote(tensors))

        EMLX.NIF.unquote(name)(unquote_splicing(args ++ extra))
        |> unquote(unwrapper)(unquote_splicing(extra))
      end
    end
  end

  defp has_device?(args) do
    Enum.any?(args, &match?({:device, _, nil}, &1))
  end

  defp tensors(args) do
    Enum.filter(args, fn {name, _, _} -> match?("tensor" <> _, Atom.to_string(name)) end)
  end
end


defmodule EMLX do
  alias EMLX.NIF, as: NIF
  use EMLX.Macro

  defguard is_tensor(device, ref) when is_reference(ref) and is_atom(device)

  @doc false
  def __emlx__, do: @emlx_function

  ## Macro callbacks

  defp normalize_device!({device, index}) when is_atom(device) and is_integer(index),
    do: {device, index}

  defp normalize_device!(device) when is_atom(device),
    do: {device, -1}

  defp normalize_device!(device),
    do: raise(ArgumentError, "expected device to be {atom, index} or atom, got: #{device}")

  defp mlx_device!(device, index) do
    case device do
      :cpu -> :cpu
      :gpu -> :gpu
      _ -> raise ArgumentError, "unknown device #{inspect(device)}"
    end
  end

  ## Creation / conversion
  def eye(size, type, device), do: eye(size, size, type, device)
  defdevice eye(m, n, type, device)
  defdevice from_blob(blob, shape, type, device)
  defdevice scalar_tensor(scalar, type, device)
  defdevice ones(shape, type, device)

  ## Manipulation
  deftensor reshape(tensor, shape)
  deftensor broadcast_to(tensor, shape)
  deftensor to_type(tensor, type)

  ## Binary ops
  deftensor add(tensorA, tensorB)
  deftensor subtract(tensorA, tensorB)
  deftensor multiply(tensorA, tensorB)
  deftensor pow(tensorA, tensorB)
  deftensor remainder(tensorA, tensorB)
  deftensor divide(tensorA, tensorB)
  deftensor atan2(tensorA, tensorB)
  deftensor bitwise_and(tensorA, tensorB)
  deftensor bitwise_or(tensorA, tensorB)
  deftensor bitwise_xor(tensorA, tensorB)
  deftensor left_shift(tensorA, tensorB)
  deftensor right_shift(tensorA, tensorB)
  deftensor min(tensorA, tensorB)
  deftensor max(tensorA, tensorB)
  deftensor equal(tensorA, tensorB)
  deftensor not_equal(tensorA, tensorB)
  deftensor greater(tensorA, tensorB)
  deftensor less(tensorA, tensorB)
  deftensor greater_equal(tensorA, tensorB)
  deftensor less_equal(tensorA, tensorB)
  deftensor logical_and(tensorA, tensorB)
  deftensor logical_or(tensorA, tensorB)
  def tensordot(tensorA, tensorB, axesA, axesB),
    do: tensordot(tensorA, tensorB, axesA, [], axesB, [])

  deftensor tensordot(tensorA, tensorB, axesA, batchA, axesB, batchB)

  ## Unary ops
  deftensor abs(tensor)
  deftensor ceil(tensor)
  deftensor conjugate(tensor)
  deftensor floor(tensor)
  deftensor negate(tensor)
  deftensor round(tensor)
  deftensor sign(tensor)
  deftensor real(tensor)
  deftensor imag(tensor)
  deftensor is_nan(tensor)
  deftensor is_infinity(tensor)
  deftensor logical_not(tensor)
  deftensor sigmoid(tensor)

  ## Aggregation
  deftensor all(tensor, axes, keep_axes)
  deftensor any(tensor, axes, keep_axes)
  deftensor sum(tensor, axes, keep_axes)
  deftensor product(tensor, axes, keep_axes)

  ## Dirty non-tensor return values
  defvalue to_blob(tensor)
  defvalue to_blob(tensor, limit)
  defvalue scalar_type(tensor)
  defvalue shape(tensor)

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("EMLX: " <> List.to_string(error))

  defp unwrap_tensor!(tagged_result, device) do
    case unwrap!(tagged_result) do
      ref when is_reference(ref) ->
        {device, ref}

      list when is_list(list) ->
        Enum.map(list, &{device, &1})

      tuple when is_tuple(tuple) ->
        tuple |> Tuple.to_list() |> Enum.map(&{device, &1}) |> List.to_tuple()
    end
  end

  defp prepare_tensors_list!(tensors_list, dev) do
    tensors =
      Enum.map(tensors_list, fn
        {^dev, ref} when is_tensor(dev, ref) ->
          ref

        {other_dev, ref} when is_tensor(other_dev, ref) ->
          raise ArgumentError, "cannot perform operation across devices #{dev} and #{other_dev}"

        bad_tensor ->
          raise ArgumentError, "expected a EMLX tensor, got: #{inspect(bad_tensor)}"
      end)

    {tensors, dev}
  end

  defp prepare_tensors!(tensors) do
    Enum.map_reduce(tensors, nil, fn
      {dev, ref}, nil when is_tensor(dev, ref) ->
        {ref, dev}

      {dev, ref}, dev when is_tensor(dev, ref) ->
        {ref, dev}

      {dev, ref}, other_dev when is_tensor(dev, ref) ->
        raise ArgumentError, "cannot perform operation across devices #{dev} and #{other_dev}"

      [{dev, ref} | _] = tensors, nil when is_tensor(dev, ref) ->
        prepare_tensors_list!(tensors, dev)

      tensors, dev when is_list(tensors) ->
        prepare_tensors_list!(tensors, dev)

      bad_tensor, _dev ->
        raise ArgumentError, "expected a EMLX tensor, got: #{inspect(bad_tensor)}"
    end)
  end

  def deallocate(tensor_ref) do
    NIF.deallocate(tensor_ref)
  end
end
