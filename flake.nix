{
  description = "Local Parakeet voice dictation with Wayland overlay";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # Runtime CLI tools justsayit shells out to.
      runtimeTools = with pkgs; [ wl-clipboard dotool wtype ];

      justsayit = pkgs.python3Packages.buildPythonApplication {
        pname = "justsayit";
        version = "0.5.3";
        pyproject = true;

        src = pkgs.lib.cleanSource ./.;

        build-system = with pkgs.python3Packages; [ hatchling ];

        dependencies = with pkgs.python3Packages; [
          sherpa-onnx
          sounddevice
          numpy
          platformdirs
          pygobject3
        ];

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
            # Point ALSA at the system config + plugins so PortAudio finds the
            # PipeWire virtual device (and thus accepts any sample rate).
            # The Nix alsa-lib's built-in config doesn't include PipeWire.
            # System ALSA config routes the default PCM through PipeWire.
            # Use the Nix pipewire plugin so its libpipewire-0.3 dep is found
            # via Nix store RPATH rather than requiring a system search.
            "--set-default" "ALSA_CONFIG_DIR" "/usr/share/alsa"
            "--set-default" "ALSA_PLUGIN_DIR" "${pkgs.pipewire}/lib/alsa-lib"
          )
        '';

        # nixpkgs sherpa-onnx installs only the module directory with no
        # .dist-info, so importlib.metadata cannot find it and the runtime
        # deps check always fails.  Skip it — the import works fine at runtime.
        dontCheckRuntimeDeps = true;

        # Tests require audio hardware and a running Wayland compositor.
        doCheck = false;
      };
    in
    {
      packages.${system} = {
        default = justsayit;
        justsayit = justsayit;
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
          wtype
        ];

        # Expose the typelibs so a bare `python` / `uv run` inside the shell
        # can import gi.repository.Gtk4LayerShell without extra setup.
        shellHook = ''
          export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0:${pkgs.gtk4-layer-shell}/lib/girepository-1.0:${pkgs.glib.out}/lib/girepository-1.0''${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
        '';
      };
    };
}
