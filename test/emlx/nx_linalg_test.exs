defmodule EMLX.Nx.LinalgTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(EMLX.Backend)
    :ok
  end

  # Note: most of these are depending on gather
  @not_implemented_yet [
    lu: 2,
    determinant: 1
  ]

  @rounding_error [
    norm: 2,
    triangular_solve: 3,
    solve: 2,
    matrix_power: 2,
    eigh: 2,
    svd: 2,
    cholesky: 1,
    least_squares: 3,
    invert: 1,
    pinv: 2
  ]

  doctest Nx.LinAlg, except: @not_implemented_yet ++ @rounding_error
end
