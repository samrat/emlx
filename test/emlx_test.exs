defmodule EMLXTest do
  use EMLX.Case
  doctest EMLX

  test "__jit__" do
    {left, right} =
      Nx.Defn.jit_apply(&{Nx.add(&1, &2), Nx.subtract(&1, &2)}, [Nx.tensor(1), 2], compiler: EMLX)

    assert_equal(left, Nx.tensor(3))
    assert_equal(right, Nx.tensor(-1))
  end
end
