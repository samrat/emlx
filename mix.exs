defmodule EMLX.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    libmlx_dir = libmlx_config().dir

    [
      app: :emlx,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # elixir_make
      make_env: %{
        "MLX_INCLUDE_DIR" => Path.join(libmlx_dir, "include"),
        "MLX_LIB_DIR" => Path.join(libmlx_dir, "lib"),
        "EMLX_VERSION" => @version,
        "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}"
      },

      # Compilers
      compilers: [:mlx, :elixir_make] ++ Mix.compilers(),
      aliases: aliases()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :inets, :ssl, :public_key, :crypto]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:elixir_make, "~> 0.6"},
      {:nx, "~> 0.9.2"}
    ]
  end

  defp aliases do
    [
      "compile.mlx": &download_and_unarchive/1
    ]
  end

  defp libmlx_config() do
    version = System.get_env("LIBMLX_VERSION", "0.21.0")

    %{
      version: version,
      base: "libmlx",
      dir: Path.join(__DIR__, "cache/libmlx-#{version}")
    }
  end

  defp download_and_unarchive(args) do
    libmlx_config = libmlx_config()

    cache_dir =
      if dir = System.get_env("LIBMLX_CACHE") do
        Path.expand(dir)
      else
        :filename.basedir(:user_cache, "libmlx")
      end

    if "--force" in args do
      File.rm_rf(libmlx_config.dir)
      File.rm_rf(cache_dir)
    end

    if File.dir?(libmlx_config.dir) do
      {:ok, []}
    else
      download_and_unarchive(cache_dir, libmlx_config)
    end
  end

  defp download_and_unarchive(cache_dir, libmlx_config) do
    File.mkdir_p!(cache_dir)
    libmlx_archive = Path.join(cache_dir, "libmlx-#{libmlx_config.version}.tar.gz")

    unless File.exists?(libmlx_archive) do
      # Download libmlx

      if {:unix, :darwin} != :os.type() do
        Mix.raise("No MLX support on non Apple Silicon machines")
      end

      url =
        "https://github.com/cocoa-xu/mlx-build/releases/download/v#{libmlx_config.version}/mlx-arm64-apple-darwin.tar.gz"

      sha256_url = "#{url}.sha256"

      download!(url, libmlx_archive)

      libmlx_archive_checksum = checksum!(libmlx_archive)

      data = download!(sha256_url)
      checksum = String.split(data, " ", parts: 2, trim: true)

      if length(checksum) != 2 do
        Mix.raise("Invalid checksum file: #{sha256_url}")
      end

      expected_checksum = hd(checksum)

      if expected_checksum != libmlx_archive_checksum do
        Mix.raise(
          "Checksum mismatch for #{libmlx_archive}. Expected: #{expected_checksum}, got: #{libmlx_archive_checksum}"
        )
      end
    end

    # Unpack libmlx and move to the target cache dir
    parent_libmlx_dir = Path.join(Path.dirname(libmlx_config.dir), "libmlx")
    File.mkdir_p!(parent_libmlx_dir)

    # Extract to the parent directory (it will be inside the libmlx directory)
    :ok = :erl_tar.extract(libmlx_archive, [:compressed, {:cwd, parent_libmlx_dir}])

    # And then rename
    File.rename!(parent_libmlx_dir, libmlx_config.dir)

    :ok
  end

  def download!(url, save_as \\ nil) do
    url_charlist = String.to_charlist(url)

    if proxy = System.get_env("HTTP_PROXY") || System.get_env("http_proxy") do
      Mix.shell().info("Using HTTP_PROXY: #{proxy}")
      %{host: host, port: port} = URI.parse(proxy)

      :httpc.set_options([{:proxy, {{String.to_charlist(host), port}, []}}])
    end

    if proxy = System.get_env("HTTPS_PROXY") || System.get_env("https_proxy") do
      Mix.shell().info("Using HTTPS_PROXY: #{proxy}")
      %{host: host, port: port} = URI.parse(proxy)
      :httpc.set_options([{:https_proxy, {{String.to_charlist(host), port}, []}}])
    end

    # https://erlef.github.io/security-wg/secure_coding_and_deployment_hardening/inets
    # TODO: This may no longer be necessary from Erlang/OTP 25.0 or later.
    https_options = [
      ssl:
        [
          verify: :verify_peer,
          customize_hostname_check: [
            match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
          ]
        ] ++ cacerts_options()
    ]

    options = [body_format: :binary]

    case :httpc.request(:get, {url_charlist, []}, https_options, options) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        if save_as do
          File.write!(save_as, body)
        end

        body

      other ->
        Mix.raise("Failed to download #{url}, reason: #{inspect(other)}")
    end
  end

  defp cacerts_options do
    cond do
      path = System.get_env("HEX_CACERTS_PATH") ->
        [cacertfile: path]

      certs = otp_cacerts() ->
        [cacerts: certs]

      true ->
        warn_no_cacerts()
        []
    end
  end

  defp otp_cacerts do
    if System.otp_release() >= "25" do
      # cacerts_get/0 raises if no certs found
      try do
        :public_key.cacerts_get()
      rescue
        _ -> nil
      end
    end
  end

  defp warn_no_cacerts do
    Mix.shell().error("""
    No certificate trust store was found.

    A certificate trust store is required in
    order to download locales for your configuration.
    Since elixir_make could not detect a system
    installed certificate trust store one of the
    following actions may be taken:

    1. Specify the location of a certificate trust store
       by configuring it in environment variable:

         export HEX_CACERTS_PATH="/path/to/cacerts.pem"

    2. Use OTP 25+ on an OS that has built-in certificate
       trust store.
    """)
  end

  defp checksum!(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        Base.encode16(:crypto.hash(:sha256, content), case: :lower)

      {:error, reason} ->
        Mix.raise(
          "Cannot read the file for checksum comparison: #{file_path}. Reason: #{inspect(reason)}"
        )
    end
  end
end
