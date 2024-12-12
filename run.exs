System.pid() |> dbg()
IO.gets("Press enter to run...")

Nx.global_default_backend(EMLX.Backend)

Nx.Defn.jit_apply(&Nx.add/2, [1, 2], compiler: EMLX)
|> dbg()
