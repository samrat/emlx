defmodule EMLX.NIF do
  @moduledoc """
  Elixir bindings for MLX array operations.
  """

  def ones(_shape, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def eye(_m, _n, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def broadcast_to(_tensor, _shape, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def scalar_type(_array) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def all(_array, _axes, _keep_dims, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def any(_array, _axes, _keep_dims, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sum(_array, _axes, _keep_dims, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def product(_array, _axes, _keep_dims, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def shape(_array) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_type(_array, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def reshape(_array, _shape, _type) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_blob(_array) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_blob(_array, _limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def from_blob(_shape, _type, _binary, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tensordot(_a, _b, _axes_a, _axes_b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def scalar_tensor(_value, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def abs(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def ceil(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def conjugate(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def floor(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def negate(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def round(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sign(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def real(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def imag(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def is_nan(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def is_infinity(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def logical_not(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sigmoid(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def add(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def subtract(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def multiply(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def pow(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def remainder(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def divide(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def atan2(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def bitwise_and(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def bitwise_or(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def bitwise_xor(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def left_shift(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def right_shift(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def min(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def max(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def equal(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def not_equal(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def greater(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def less(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def greater_equal(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def less_equal(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def logical_and(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def logical_or(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def deallocate(_ref) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @on_load :load_nifs
  def load_nifs do
    path = :filename.join(:code.priv_dir(:emlx), ~c"libemlx")
    :erlang.load_nif(path, 0)
  end
end
