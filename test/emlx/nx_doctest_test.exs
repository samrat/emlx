defmodule EMLX.Nx.DoctestTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(EMLX.Backend)
    :ok
  end

  @not_implemented_yet [
    # not supported yet
    reverse: 2
  ]

  @rounding_error [
    erfc: 1,
    expm1: 1,
    atan: 1,
    sigmoid: 1,
    round: 1,
    asinh: 1,
    asin: 1,
    tan: 1,
    cos: 1,
    standard_deviation: 2,
    cosh: 1,
    log10: 1,
    acos: 1,
    covariance: 3
  ]

  @to_be_fixed [
    :moduledoc,
    # window_* do not support window_dilations yet
    window_sum: 3,
    window_max: 3,
    window_min: 3,
    window_product: 3,
    window_mean: 3,
    # integer types not supported, and complex types not supported
    # complex can use the definition Torchx uses
    conv: 3,
    # missing support for inner padding
    pad: 3,
    # MLX sorts NaNs lowest, Nx sorts them highest
    argmin: 2,
    argmax: 2,
    argsort: 2,
    # Missing support for window dilations and for tie_break: :high
    window_scatter_max: 5,
    window_scatter_min: 5
  ]

  @not_supported [
    # Does not support f8 (yet?)
    tensor: 2,
    # Does not support u2 (yet?)
    bit_size: 1,
    # f64 not supported
    from_binary: 3,
    # f64 not supported
    iota: 2,
    # f64 not supported
    atan2: 2,
    # f64 not supported
    as_type: 2,
    reduce: 4,
    window_reduce: 5,
    population_count: 1,
    count_leading_zeros: 1,
    sort: 2
  ]

  doctest Nx, except: @not_implemented_yet ++ @rounding_error ++ @not_supported ++ @to_be_fixed
end
