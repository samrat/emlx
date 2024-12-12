System.pid() |> dbg()
IO.gets("Press enter to run...")

Nx.global_default_backend(EMLX.Backend)

Nx.Defn.jit_apply(&{Nx.add(&1, &2), Nx.subtract(&1, &2)}, [Nx.tensor(1), Nx.tensor(2)], compiler: EMLX)
|> dbg()
