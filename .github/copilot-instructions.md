# Stable Diffusion Web UI - AI Agent Instructions

## Architecture Overview

This is a **Gradio-based web interface** for Stable Diffusion models. The codebase follows a **monolithic architecture** with clear separation between UI (`modules/ui*.py`), processing logic (`modules/processing.py`, `modules/txt2img.py`, `modules/img2img.py`), and model integration (`modules/sd_*.py`).

### Key Components

- **Entry Points**: `webui.py` (UI mode) and `launch.py` (environment setup)
- **Core Processing**: `modules/processing.py` contains `StableDiffusionProcessingTxt2Img` and `StableDiffusionProcessingImg2Img` classes that orchestrate generation
- **Model Hijacking**: `modules/sd_hijack*.py` patches PyTorch model internals for custom behavior (attention mechanisms, embeddings, optimizations)
- **Scripts System**: `modules/scripts.py` provides plugin architecture - scripts define `title()`, `ui()`, and `run()` methods
- **Extensions**: Located in `extensions-builtin/` (bundled) and `extensions/` (user-installed), loaded via `modules/extensions.py`
- **API Layer**: `modules/api/api.py` exposes FastAPI endpoints wrapping the same processing functions

### Critical Data Flow

1. User input → Gradio UI (`modules/ui.py`) → Processing classes (`modules/processing.py`)
2. Processing applies scripts, extra networks (LoRAs/hypernetworks), and prompt parsing
3. Model hijacking intercepts forward passes to inject custom attention and embeddings
4. Generated images go through postprocessing pipeline, then returned to UI/API

## Development Workflows

### Running Locally

```powershell
# Windows (primary platform)
.\webui.bat                    # Full UI with automatic venv setup
python launch.py               # Direct launch (assumes environment ready)
python launch.py --api-only    # API server only, no Gradio UI
```

**Important Launch Flags** (see `modules/cmd_args.py`):
- `--xformers`: Enable memory-efficient attention (requires xformers package)
- `--medvram` / `--lowvram`: VRAM optimizations for low-end GPUs
- `--api`: Enable API alongside UI
- `--skip-prepare-environment`: Skip dependency checks (faster iteration)

### Testing

```powershell
# Run pytest suite (tests use API endpoints)
pytest test/

# Test specific modules
pytest test/test_txt2img.py
```

Tests in `test/` verify API functionality by starting the server with `--test-server` flag.

### Extension Development

Extensions use **callback hooks** defined in `modules/script_callbacks.py`:

```python
# Example extension structure (see extensions-builtin/Lora/)
import modules.script_callbacks as script_callbacks

def on_ui_tabs():
    # Add custom Gradio tabs
    pass

script_callbacks.on_ui_tabs(on_ui_tabs)
```

**Key Callbacks**:
- `on_before_ui()` / `on_after_ui()`: UI initialization
- `on_model_loaded()`: Model checkpoint changes
- `on_script_unloaded()`: Cleanup
- `on_infotext_pasted()`: Parse generation parameters from images

### Script Development

Scripts inherit from `modules.scripts.Script`:

```python
class Script(scripts.Script):
    def title(self):
        return "Script Name"
    
    def ui(self, is_img2img):
        # Return Gradio components
        return [component1, component2]
    
    def run(self, p, component1_value, component2_value):
        # Modify processing parameters (p)
        # p.prompt, p.steps, p.width, etc.
        proc = process_images(p)
        return proc
```

See `scripts/xyz_grid.py` for a comprehensive example.

## Project-Specific Patterns

### Model Hijacking System

**Critical**: The codebase **patches PyTorch model forward passes** to inject custom behavior. Do not bypass hijacks when modifying model code.

- `modules/sd_hijack.py`: Main orchestration
- `modules/sd_hijack_clip.py`: Text encoder patches for textual inversion embeddings
- `modules/sd_hijack_unet.py`: UNet patches for hypernetworks/LoRAs
- `modules/sd_hijack_optimizations.py`: Memory-efficient attention implementations

### Shared State Management

`modules/shared.py` provides **global singletons**:

```python
import modules.shared as shared

shared.sd_model          # Current loaded model
shared.opts             # User settings (Options object)
shared.state            # Generation state (progress, interruption)
shared.cmd_opts         # CLI arguments
```

**Never reassign** `shared.sd_model` directly - use `modules/sd_models.py::reload_model_weights()`.

### Extra Networks (LoRAs, Hypernetworks, Textual Inversion)

Referenced in prompts via special syntax: `<lora:name:weight>`, `<hypernet:name:weight>`, `<embedding_name>`

- Parsed by `modules/extra_networks.py::parse_prompts()`
- Applied during processing via `activate()` callbacks
- Each type has dedicated handler: `extra_networks_lora.py`, `extra_networks_hypernet.py`, etc.

### Infotext System

Generation parameters embedded in PNG chunks:

```python
from modules import infotext_utils

# Parsing parameters from images
params = infotext_utils.parse_generation_parameters(infotext_string)

# Creating infotext
p.extra_generation_params["Custom Param"] = value  # Adds to output
```

Used for "Send to txt2img" buttons and reproducibility.

### Settings and Options

Settings defined in `modules/shared_options.py` using `OptionInfo`:

```python
shared.options_templates.update(shared.options_section(('section', "Section Name"), {
    "option_key": shared.OptionInfo(default, "Label", gr.Component, extra_kwargs),
}))
```

Accessed via `shared.opts.option_key`. Changes trigger UI refresh.

## Common Gotchas

1. **Gradio component IDs**: Use `elem_id` for JavaScript/CSS hooks, stored in `modules/ui_components.py`
2. **Processing parameter validation**: Always call `processing.fix_seed(p)` before generation loops
3. **Device management**: Use `modules.devices.torch_gc()` to force garbage collection, `devices.device` for current device
4. **Image format**: Processing uses PIL Images, latents are PyTorch tensors - convert carefully
5. **Timer imports**: `modules/timer.py::startup_timer` tracks initialization - record checkpoints for profiling

## API Integration

FastAPI endpoints mirror UI functionality:

```python
# Text-to-image (POST /sdapi/v1/txt2img)
{
    "prompt": "...",
    "negative_prompt": "...",
    "steps": 20,
    "width": 512,
    "height": 512,
    # ... see modules/api/models.py for schema
}
```

Scripts accessible via API require `api_info` attribute. See `modules/api/api.py::script_name_to_index()`.

## File Locations

- **Models**: `models/Stable-diffusion/*.safetensors` (checkpoints), `models/VAE/`, `models/Lora/`
- **Outputs**: `outputs/txt2img-images/`, `outputs/img2img-images/`
- **User Data**: `embeddings/`, `extensions/`, `styles.csv` (prompt templates)
- **Config**: `config.json` (UI settings), `ui-config.json` (component defaults)

## Key Dependencies

- **gradio==3.41.2**: UI framework (pinned version - breaking changes frequent)
- **torch**: PyTorch (version varies by CUDA support)
- **transformers==4.30.2**: Hugging Face models (text encoders)
- **safetensors**: Preferred checkpoint format over pickle

When modifying dependencies, test with `--skip-prepare-environment` to avoid reinstalls.
