{
  description = "Local Parakeet voice dictation with Wayland overlay";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # Runtime CLI tools justsayit shells out to.
      runtimeTools = with pkgs; [ wl-clipboard dotool ];

      # nixpkgs ships llama-cpp-python 0.3.16, which predates the qwen35 / gemma4
      # architectures.  Override the source to 0.3.20 (latest on PyPI) so all
      # current models are supported while reusing the nixpkgs build recipe.
      llama-cpp-python-new = pkgs.python3Packages.llama-cpp-python.overridePythonAttrs (_old: {
        version = "0.3.20";
        # PyPI sdist bundles llama.cpp inline (no submodules needed).
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/45/95/c69c47c9c8dda97f712f5864688d13a22b0159aa9adae91a69067a728532/llama_cpp_python-0.3.20.tar.gz";
          hash = "sha256-cPAbfZFdMcYX3GZhCjMsstUc3oTIsVLQmjUiBjI2FvU=";
        };
        # The nixpkgs patch is a macOS Metal test fix against 0.3.16's llama.cpp;
        # it doesn't apply to 0.3.20 and is irrelevant on Linux.
        patches = [];
      });

      # llama-cpp-python 0.3.20 rebuilt with Vulkan support for GPU inference.
      llama-cpp-python-vulkan = llama-cpp-python-new.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ (with pkgs; [
          shaderc   # provides glslc, needed at compile time to build SPIR-V shaders
        ]);
        buildInputs = (old.buildInputs or []) ++ (with pkgs; [
          vulkan-headers
          vulkan-loader
        ]);
        env = (old.env or {}) // {
          CMAKE_ARGS = ((old.env or {}).CMAKE_ARGS or "") + " -DGGML_VULKAN=1";
        };
      });

      # Vulkan ICDs (driver descriptors) from nixpkgs mesa.  Each JSON points at
      # an absolute /nix/store path for its .so, so the sandboxed nixpkgs
      # vulkan-loader can actually load it — unlike system ICDs at
      # /usr/share/vulkan/icd.d/, whose relative library_path fields only
      # resolve against the host distro's /usr/lib.  Covers AMD (radv), Intel
      # (anv), Nouveau, lavapipe (CPU fallback), virtio, etc.  NVIDIA
      # proprietary driver is not covered — those users need nixGL.
      vulkanICDs = pkgs.runCommand "vulkan-icds" {} ''
        mkdir -p $out
        cp ${pkgs.mesa}/share/vulkan/icd.d/*.json $out/
      '';

      mkJustsayit = { withLlm ? false, withVulkan ? false, llamaCppPython ? llama-cpp-python-new }: pkgs.python3Packages.buildPythonApplication {
        pname = "justsayit";
        version = "0.24.2";
        pyproject = true;

        src = pkgs.lib.cleanSource ./.;

        build-system = with pkgs.python3Packages; [ hatchling ];

        dependencies = with pkgs.python3Packages; [
          sherpa-onnx
          sounddevice
          numpy
          platformdirs
          pygobject3
        ] ++ pkgs.lib.optionals withLlm [ llamaCppPython ];

        # gobject-introspection scans buildInputs for typelibs at build time;
        # wrapGAppsHook4 populates gappsWrapperArgs with GI_TYPELIB_PATH,
        # XDG_DATA_DIRS, etc.
        nativeBuildInputs = with pkgs; [
          gobject-introspection
          wrapGAppsHook4
        ];

        # GI libraries whose typelibs are needed at runtime.
        # pipewire gives sounddevice/PortAudio the PipeWire ALSA plugin so
        # audio device negotiation works the same as with the system install.
        buildInputs = with pkgs; [
          glib
          gtk4
          gtk4-layer-shell
          pipewire
        ];

        # buildPythonApplication wraps executables via wrapPythonPrograms, which
        # reads makeWrapperArgs.  wrapGAppsHook4 would wrap them a second time,
        # so we disable its wrapping step (dontWrapGApps) and instead fold
        # gappsWrapperArgs into makeWrapperArgs so both happen in one pass.
        dontWrapGApps = true;

        preFixup = ''
          makeWrapperArgs+=(
            "''${gappsWrapperArgs[@]}"
            "--suffix" "PATH" ":" "${pkgs.lib.makeBinPath runtimeTools}"
            # gtk4-layer-shell must be loaded before GTK initialises.
            # Normally the app re-execs itself with LD_PRELOAD to achieve
            # this, but that re-exec uses sys.argv[0] which the Nix Python
            # wrapper has already replaced with the ELF binary path —
            # causing Python to try to execute the ELF as source.
            # Instead, inject the preload here and skip the re-exec.
            "--prefix" "LD_PRELOAD" ":" "${pkgs.gtk4-layer-shell}/lib/libgtk4-layer-shell.so"
            "--set" "_JUSTSAYIT_PRELOADED" "1"
            # Use the Nix pipewire ALSA plugin so PortAudio can route through
            # PipeWire. Don't override ALSA_CONFIG_DIR - it breaks NixOS where
            # /usr/share/alsa doesn't exist; let each system use its default.
            "--set-default" "ALSA_PLUGIN_DIR" "${pkgs.pipewire}/lib/alsa-lib"
            ${pkgs.lib.optionalString withVulkan ''
            # Add nixpkgs mesa ICDs to the vulkan-loader's search so GPU
            # inference works on non-NixOS hosts (whose system ICD JSONs have
            # relative library_paths the sandboxed loader can't resolve).
            # VK_ADD_DRIVER_FILES *appends* rather than replaces, so NixOS
            # users keep their NVIDIA / system ICDs from /run/opengl-driver.
            "--suffix" "VK_ADD_DRIVER_FILES" ":" "$(echo ${vulkanICDs}/*.json | tr ' ' ':')"
            ''}
          )
        '';

        # nixpkgs sherpa-onnx installs only the module directory with no
        # .dist-info, so importlib.metadata cannot find it and the runtime
        # deps check always fails.  Skip it — the import works fine at runtime.
        dontCheckRuntimeDeps = true;

        # Tests require audio hardware and a running Wayland compositor.
        doCheck = false;
      };
      justsayit = mkJustsayit {};
    in
    {
      packages.${system} = {
        default = justsayit;
        justsayit = justsayit;
        with-llm = mkJustsayit { withLlm = true; };
        with-llm-vulkan = mkJustsayit { withLlm = true; withVulkan = true; llamaCppPython = llama-cpp-python-vulkan; };
        # Exposed for debugging (nm, readelf on libllama.so / libggml-vulkan.so).
        llama-cpp-python-vulkan = llama-cpp-python-vulkan;
      };

      # Development shell: system deps for PyGObject + the uv workflow.
      # Usage: nix develop, then: uv sync --system-site-packages
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          gobject-introspection
          pkg-config
        ];

        packages = with pkgs; [
          uv
          python3
          gtk4
          gtk4-layer-shell
          glib
          wl-clipboard
          dotool
        ];

        # Expose the typelibs so a bare `python` / `uv run` inside the shell
        # can import gi.repository.Gtk4LayerShell without extra setup.
        shellHook = ''
          export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0:${pkgs.gtk4-layer-shell}/lib/girepository-1.0:${pkgs.glib.out}/lib/girepository-1.0''${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
        '';
      };
    };
}
