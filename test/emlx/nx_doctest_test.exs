defmodule EMLX.Nx.DoctestTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(EMLX.Backend)
    :ok
  end

  @not_implemented_yet [
    indexed_add: 4,
    indexed_put: 4,
    # put_diagonal depends on indexed_put
    put_diagonal: 3,
    # take_diagonal depends on gather
    take_diagonal: 2,
    # make_diagonal depends on indexed_put
    make_diagonal: 2,
    # mode depends on indexed_add
    mode: 2,
    pad: 3,
    conv: 3,
    # dot does not support integer types
    dot: 2,
    dot: 4,
    dot: 6,
    window_scatter_min: 5,
    window_scatter_max: 5,
    reverse: 2,
    take: 3,
    slice: 4,
    gather: 3,
    reflect: 2
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
    log10: 1
  ]

  @to_be_fixed [
    :moduledoc,
    argmin: 2,
    argmax: 2,
    argsort: 2,
    sort: 2,
    ifft2: 2,
    fft2: 2,
    real: 1,
    imag: 1,
    acosh: 1,
    acos: 1,
    phase: 1,
    remainder: 2,
    complex: 2,
    is_infinity: 1,
    window_sum: 3,
    window_max: 3,
    window_min: 3,
    window_product: 3,
    window_mean: 3,
    all_close: 3,
    conjugate: 1,
    bitwise_not: 1,
    clip: 3,
    covariance: 3
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
    count_leading_zeros: 1
  ]

  doctest Nx, except: @not_implemented_yet ++ @rounding_error ++ @not_supported ++ @to_be_fixed
end
