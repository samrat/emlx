# EMLX

**TODO**

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `emlx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:emlx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/emlx>.


### MLX binaries

EMLX relies on the [MLX](https://github.com/ml-explore/mlx) library to function, and currently EMLX will download precompiled builds from [mlx-build](https://github.com/cocoa-xu/mlx-build). 

Compiling from source and using customized precompiled binaries will be supported soon.

#### Using precompiled binaries

While the default configuration should be suitable for most cases, there is however a number of environment variables that you may want to use in order to customize the variant of MLX binary.

The binaries are always downloaded to match the current configuration, so you should set the environment variables in .bash_profile or a similar configuration file so you don't need to export it in every shell session.

##### `LIBMLX_VERSION`

The version of the MLX binary to download. By default EMLX will always use the latest version possible.

##### `LIBMLX_ENABLE_JIT`

Defaults to `false`. 

Using JIT compilation for Metal kernels when set to `true`.

##### `LIBMLX_ENABLE_DEBUG`

Defaults to `false`. 

Enhance metal debug workflow by enabling debug information in the Metal shaders when set to `true`.

##### `LIBMLX_CACHE`

The directory to store the downloaded and built archives in. Defaults to the standard cache location for the given operating system.
