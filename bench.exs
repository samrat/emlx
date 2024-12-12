
Mix.install [{:emlx, path: __DIR__}, :benchee, :exla]

System.pid() |> dbg()
IO.gets("Press enter to run...")


defmodule ToBench do
  def run(backend, compiler) do
    prev_backend = Nx.default_backend(backend)
    fun = fn x, y ->
      Enum.reduce(1..10, x, fn x, acc ->
        Nx.dot(x, acc)
        |> Nx.add(y)
      end)
    end


    Nx.Defn.jit_apply(fun, [Nx.iota({10, 10}, type: :f32, backend: backend), Nx.iota({10, 10}, type: :f32, backend: backend)], compiler: compiler)
    |> tap(fn _ -> Nx.default_backend(prev_backend) end)
  end
end


# ToBench.run(EMLX.Backend, Nx.Defn.Evaluator) |> dbg()
# ToBench.run(EMLX.Backend, EMLX) |> dbg()
# ToBench.run({EMLX.Backend, device: :gpu}, EMLX) |> dbg()

Benchee.run(%{
  "EMLX (evaluator)" => fn -> ToBench.run(EMLX.Backend, Nx.Defn.Evaluator) end,
  "EMLX" => fn -> ToBench.run(EMLX.Backend, EMLX) end,
  "EMLX gpu" => fn -> ToBench.run({EMLX.Backend, device: :gpu}, EMLX) end,
  "EXLA (evaluator)" => fn -> ToBench.run(EXLA.Backend, Nx.Defn.Evaluator) end,
  "EXLA" => fn -> ToBench.run(EXLA.Backend, EXLA) end
})
