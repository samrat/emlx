defmodule EMLX.Application do
  use Application

  def start(_type, _args) do
    children = [
      {EMLX.NIF.CallEvaluator,
       nif_module: EMLX.NIF, process_options: [name: EMLX.NIF.CallEvaluator]}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: EMLX.Supervisor)
  end
end
