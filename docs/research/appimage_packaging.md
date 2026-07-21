# Research: AppImage Packaging for PySide6 + Embedded Qdrant

**Date:** 2026-07-15
**Ticket:** #45
**Status:** Complete

## Problem Statement

Package the RAG system (Python 3.12, PySide6 UI, embedded Qdrant, SQLite, sentence-transformers models) as a self-contained Linux AppImage with a desktop entry, built via GitHub Actions.

Key constraints:
- Large binary dependencies: PyTorch/sentence-transformers (~2 GB), Qdrant embedded engine
- Mutable user data: `qdrant_data/`, `rag_system.db`, `config/settings.yaml`
- Must resolve paths correctly in both dev mode and packaged mode
- Target: x86_64 Linux (glibc 2.31+, i.e., Ubuntu 20.04+)

---

## Tool Comparison

### 1. PyInstaller + appimagetool

**How it works:** PyInstaller bundles Python + all dependencies into a single directory (`--onedir`). Then `appimagetool` wraps that directory into an AppImage.

| Pros | Cons |
|------|------|
| Mature, well-documented for Python | Two-step process (PyInstaller then appimagetool) |
| Handles hidden imports, data files | PyInstaller hooks for PyTorch/transformers can be fragile |
| Good PySide6 support via hooks | Large output (~2-3 GB for ML stack) |
| Full control over what's included | Need to manually handle `.so` library bundling edge cases |

**Verdict:** Best balance of maturity and control for Python apps. The two-step process is straightforward to automate.

### 2. linuxdeploy + linuxdeploy-plugin-python

**How it works:** linuxdeploy creates an AppDir structure and bundles system libraries. The Python plugin embeds a CPython interpreter and pip-installs dependencies.

| Pros | Cons |
|------|------|
| Proper AppImage workflow | Python plugin is less maintained than PyInstaller |
| Handles library dependencies automatically | Harder to debug dependency resolution failures |
| Desktop integration built-in | Less community knowledge for complex Python stacks |
| Can bundle system `.so` files cleanly | pip install inside AppDir can miss compiled extensions |

**Verdict:** Better suited for C/C++ apps with a thin Python layer. Overkill complexity for a pure-Python project.

### 3. appimage-builder (AppImageCraft)

**How it works:** YAML-based recipe that uses apt to resolve dependencies, builds an AppDir, and produces the AppImage.

| Pros | Cons |
|------|------|
| Declarative YAML config | Relies on apt packages (ties you to Debian/Ubuntu) |
| Handles complex dependency trees | Poor fit for pip/uv-managed Python dependencies |
| Built-in testing framework | Less flexible for custom build steps |
| Good for system-level deps | Documentation gaps for Python ML stacks |

**Verdict:** Designed for packaging existing distro packages. Poor fit for a uv-managed Python project with pip dependencies.

### 4. Briefcase (BeeWare)

**How it works:** Python packaging tool that targets multiple platforms. For Linux, it produces AppImages using a pre-built Python runtime.

| Pros | Cons |
|------|------|
| Python-native, understands pyproject.toml | Opinionated project structure required |
| Cross-platform (macOS, Windows, Linux) | Less control over bundling details |
| Handles PySide6/Qt well | Struggles with large native deps (PyTorch) |
| Clean AppImage output | Would require restructuring the project |

**Verdict:** Good for simple PySide6 apps but struggles with heavy ML dependencies and the project restructuring cost is high.

### 5. Nuitka + appimagetool

**How it works:** Nuitka compiles Python to C, producing a standalone binary. Wrap with appimagetool.

| Pros | Cons |
|------|------|
| Faster startup, smaller base size | Very long compilation times (30+ min for this project) |
| Better IP protection (compiled) | Compatibility issues with dynamic imports (LangChain, transformers) |
| Single binary output | Debugging compiled output is painful |
| Can reduce final size | Plugin system for packages is less mature than PyInstaller |

**Verdict:** Compilation issues with LangChain's dynamic plugin system and sentence-transformers make this impractical.

---

## Recommendation: PyInstaller + appimagetool

**Rationale:**
1. PyInstaller has the strongest ecosystem for complex Python apps (PyTorch hooks, Qt hooks)
2. Two-step process (freeze then wrap) is simple to debug independently
3. Largest community knowledge base for troubleshooting
4. GitHub Actions integration is well-documented
5. Compatible with all project dependencies (LangChain, sentence-transformers, PySide6, qdrant-client)

---

## Runtime Path Resolution Strategy

### The Problem

In dev mode, paths are relative to the project root:
```
./qdrant_data/
./rag_system.db
./config/settings.yaml
./data/  (monitored folder)
```

In an AppImage, the filesystem is read-only (squashfs mounted at `$APPDIR`). User data must live outside the image.

### The Solution: XDG Base Directories

Follow the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/):

| Data type | XDG variable | Default path | Contents |
|-----------|-------------|--------------|----------|
| User data | `$XDG_DATA_HOME` | `~/.local/share/rag-system/` | `qdrant_data/`, `rag_system.db` |
| Config | `$XDG_CONFIG_HOME` | `~/.config/rag-system/` | `settings.yaml` |
| Monitored docs | User-chosen | `~/Documents/rag-system/` | Ingested files |
| Cache (models) | `$XDG_CACHE_HOME` | `~/.cache/rag-system/` | sentence-transformers models |

### Implementation: Path Resolver Module

Add a `src/core/paths.py` module:

```python
"""Resolve data paths for both dev and packaged (AppImage) modes."""
import os
import sys
from pathlib import Path


def is_packaged() -> bool:
    """Detect if running inside an AppImage."""
    return "APPIMAGE" in os.environ


def _xdg_path(env_var: str, default_subdir: str) -> Path:
    base = os.environ.get(env_var, Path.home() / default_subdir)
    return Path(base) / "rag-system"


def project_root() -> Path:
    """In dev: repo root. In AppImage: $APPDIR/usr/share/rag-system/."""
    if is_packaged():
        return Path(os.environ["APPDIR"]) / "usr" / "share" / "rag-system"
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Mutable user data (qdrant_data/, rag_system.db)."""
    if is_packaged():
        p = _xdg_path("XDG_DATA_HOME", ".local/share")
    else:
        p = project_root()
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_dir() -> Path:
    """Configuration files (settings.yaml)."""
    if is_packaged():
        p = _xdg_path("XDG_CONFIG_HOME", ".config")
    else:
        p = project_root() / "config"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    """Cache (downloaded ML models)."""
    if is_packaged():
        p = _xdg_path("XDG_CACHE_HOME", ".cache")
    else:
        p = project_root() / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def documents_dir() -> Path:
    """Default monitored folder for document ingestion."""
    if is_packaged():
        p = Path.home() / "Documents" / "rag-system"
    else:
        p = project_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p
```

### Migration of Existing Path References

Current `config/settings.yaml` has relative paths:
```yaml
storage:
  qdrant_path: ./qdrant_data
  sqlite_path: ./rag_system.db
knowledge_base:
  monitored_folder: ./data
```

The config loader (`src/core/config.py`) should resolve these relative to `data_dir()` and `documents_dir()` when packaged. On first launch in AppImage mode, copy the bundled default `settings.yaml` to `config_dir()` if not already present.

---

## Desktop Entry and Icon

### AppDir Structure

```
AppDir/
├── AppRun                          # Entry point script
├── rag-system.desktop              # Desktop entry
├── rag-system.svg                  # App icon (scalable)
├── usr/
│   ├── bin/
│   │   └── rag-system              # Symlink to actual binary
│   ├── lib/
│   │   └── python3.12/             # PyInstaller output
│   ├── share/
│   │   ├── rag-system/
│   │   │   └── config/
│   │   │       └── settings.yaml   # Default config (read-only)
│   │   ├── applications/
│   │   │   └── rag-system.desktop
│   │   └── icons/
│   │       └── hicolor/
│   │           └── scalable/
│   │               └── apps/
│   │                   └── rag-system.svg
│   └── ...
```

### Desktop Entry File

```ini
[Desktop Entry]
Name=RAG System
Comment=Local personal knowledge base with adaptive retrieval
Exec=rag-system
Icon=rag-system
Type=Application
Categories=Office;Science;Education;
Keywords=RAG;knowledge;search;AI;
StartupWMClass=rag-system
```

### AppRun Script

```bash
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export APPDIR="$HERE"
export PATH="$HERE/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HERE/usr/lib:$LD_LIBRARY_PATH"

# Set model cache to avoid re-downloading inside AppImage
export SENTENCE_TRANSFORMERS_HOME="${XDG_CACHE_HOME:-$HOME/.cache}/rag-system/models"
export TRANSFORMERS_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/rag-system/models"

exec "$HERE/usr/bin/rag-system" "$@"
```

---

## GitHub Actions Workflow

```yaml
name: Build AppImage

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build-appimage:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            libfuse2 \
            libxkbcommon0 \
            libxcb-cursor0 \
            libegl1 \
            libgl1-mesa-glx

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install project dependencies
        run: |
          uv sync --no-dev
          uv pip install pyinstaller

      - name: Build with PyInstaller
        run: |
          uv run pyinstaller \
            --name rag-system \
            --onedir \
            --windowed \
            --add-data "config/settings.yaml:config" \
            --hidden-import=sentence_transformers \
            --hidden-import=torch \
            --hidden-import=qdrant_client \
            --hidden-import=PySide6 \
            --collect-all sentence_transformers \
            --collect-all qdrant_client \
            src/main.py

      - name: Prepare AppDir
        run: |
          mkdir -p AppDir/usr/bin
          mkdir -p AppDir/usr/share/rag-system/config
          mkdir -p AppDir/usr/share/applications
          mkdir -p AppDir/usr/share/icons/hicolor/scalable/apps

          # Copy PyInstaller output
          cp -r dist/rag-system/* AppDir/usr/bin/

          # Copy default config
          cp config/settings.yaml AppDir/usr/share/rag-system/config/

          # Desktop entry and icon
          cp packaging/rag-system.desktop AppDir/
          cp packaging/rag-system.desktop AppDir/usr/share/applications/
          cp packaging/rag-system.svg AppDir/
          cp packaging/rag-system.svg AppDir/usr/share/icons/hicolor/scalable/apps/

          # AppRun
          cp packaging/AppRun AppDir/
          chmod +x AppDir/AppRun

      - name: Download appimagetool
        run: |
          wget -q https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
          chmod +x appimagetool-x86_64.AppImage

      - name: Build AppImage
        run: |
          ARCH=x86_64 ./appimagetool-x86_64.AppImage AppDir/ RAG_System-x86_64.AppImage

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: appimage
          path: RAG_System-x86_64.AppImage

      - name: Attach to release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: RAG_System-x86_64.AppImage
```

---

## Size Optimization Notes

The full bundle with PyTorch + sentence-transformers will be ~2-3 GB. Strategies to reduce:

1. **PyTorch CPU-only:** Use `torch+cpu` wheel (saves ~1 GB over CUDA build)
2. **ONNX Runtime:** Convert embedding model to ONNX, replace sentence-transformers with onnxruntime (~200 MB vs ~1.5 GB)
3. **Exclude unused torch modules:** PyInstaller `--exclude-module` for `torch.cuda`, `torch.distributed`
4. **UPX compression:** appimagetool supports `--comp zstd` for squashfs compression (30-40% reduction)

### Recommended Phase 1 Target

Ship with PyTorch CPU-only. The AppImage will be ~1.8 GB compressed. Optimize to ONNX in a follow-up ticket if size becomes a user concern.

---

## First-Run Experience

On first launch from AppImage:

1. Detect no config at `~/.config/rag-system/settings.yaml`
2. Copy bundled default from `$APPDIR/usr/share/rag-system/config/settings.yaml`
3. Create `~/.local/share/rag-system/` for `qdrant_data/` and `rag_system.db`
4. Create `~/Documents/rag-system/` as the default monitored folder
5. Download embedding model to `~/.cache/rag-system/models/` (one-time, ~1.3 GB)

Display a first-run dialog (PySide6) explaining data locations and model download.

---

## Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Packaging tool | PyInstaller + appimagetool | Most mature for complex Python apps |
| User data location | XDG dirs (`~/.local/share/rag-system/`) | Linux standard, respects user config |
| Config location | `~/.config/rag-system/` | XDG standard |
| Model cache | `~/.cache/rag-system/models/` | Standard, cleanable |
| CI runner | ubuntu-22.04 | glibc 2.35, compatible with most targets |
| Compression | zstd squashfs | Best ratio for large binaries |
| Size strategy | CPU-only PyTorch first, ONNX later | Ship fast, optimize incrementally |

---

## Open Questions for Implementation

1. Should the embedding model be bundled in the AppImage (larger image, no first-run download) or downloaded on first run (smaller image, requires internet)?
2. Do we want auto-update via AppImageUpdate delta mechanisms?
3. Should we offer a "portable mode" flag (`--portable`) that keeps all data next to the AppImage?
