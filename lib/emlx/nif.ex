defmodule EMLX.NIF do
  @moduledoc """
  Elixir bindings for MLX array operations.
  """

  def item(_tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def slice(_tensor, _starts, _stops, _strides, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def squeeze(_tensor, _axes, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def eval(_tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def ones(_shape, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def full(_value, _shape, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def arange(_start, _stop, _step, _integer?, _device) do
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

  def argmax(_array, _axes, _keep_dims, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def argmin(_array, _axes, _keep_dims, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def cumulative_sum(_array, _axis, _reverse, _inclusive, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def cumulative_product(_array, _axis, _reverse, _inclusive, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def cumulative_max(_array, _axis, _reverse, _inclusive, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def cumulative_min(_array, _axis, _reverse, _inclusive, _device) do
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

  def tensordot(_a, _b, _axesA, _axesB, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def transpose(_tensor, _axes, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def scalar_tensor(_value, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def stack(_tensors, _axis, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def concatenate(_tensors, _axis, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def where(_condition, _on_true, _on_false, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def concatenate(_tensors, _axis, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def take_along_axis(_tensor, _indices, _axis, _device) do
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

  def asin(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def asinh(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def acos(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def acosh(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def atan(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def atanh(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def cos(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def cosh(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def erf(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def erf_inv(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def exp(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def expm1(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def log(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def log1p(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def rsqrt(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sin(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sinh(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sqrt(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tan(_tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tanh(_tensor, _device) do
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

  def bitwise_not(_a, _device) do
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

  def quotient(_a, _b, _device) do
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

  def logical_xor(_a, _b, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def fft(_a, _n, _axis, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def ifft(_a, _n, _axis, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def fft2(_a, _s, _axes, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def ifft2(_a, _s, _axes, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def allclose(_a, _b, _atol, _rtol, _equal_nan, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def isclose(_a, _b, _atol, _rtol, _equal_nan, _device) do
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
