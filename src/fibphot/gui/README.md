# fibphot GUI

Launch locally after installing the GUI dependencies:

```bash
pip install -e .[gui]
fibphot-gui
```

or without the entry point:

```bash
python -m fibphot.gui.app
```

The GUI can load `.doric` files and fibphot `.h5/.hdf5` state files, build a reversible processing pipeline, run analyses, overlay detected peaks, save/load pipeline JSON, and export processed state/results.

## Extending the GUI

The GUI is registry-driven. Add new processing stages or analyses by registering them with `STAGE_REGISTRY` or `ANALYSIS_REGISTRY`; you do not need to edit `app.py`.

```python
from fibphot.gui import STAGE_REGISTRY, ParameterSpec

STAGE_REGISTRY.register(
    "MyStage",
    MyStage,
    label="My new stage",
    group="Custom",
    parameters={
        "channels": ParameterSpec("all", kind="channels"),
        "strength": ParameterSpec(1.0, kind="float", minimum=0, step=0.1),
    },
)
```

Registered parameters are automatically rendered as GUI controls and serialised in pipeline JSON.

## Interface notes

- The sidebar width control lives in the **Settings** panel.
- The trace viewer uses an interactive Bokeh plot with pan, wheel zoom, box zoom, reset, hover and save tools.
- The default trace renderer is SVG for crisp vector traces; switch to Canvas in **Settings** for very dense recordings.
- Trace and results panels remain vertically resizable.

## Output directory

The **Session** panel contains an output-directory field.  By default this is
set to the directory from which `fibphot-gui` was launched.  Relative paths for
pipeline JSON files and exports are resolved under this directory.  Absolute
paths are preserved.

For example, with output directory `/tmp/project`:

- `pipeline.json` is saved as `/tmp/project/pipeline.json`.
- export folder `fibphot_export` writes to `/tmp/project/fibphot_export`.
- export folder `/tmp/other_export` writes to `/tmp/other_export`.

## Channel-aware controls and colours

When a recording or processed batch session is loaded, any registered parameter
with `kind="channel"` is rendered as a dropdown populated from the available
channel names.  Parameters with `kind="channels"` are rendered as a multi-select
control and default to all non-isosbestic channels.  This avoids hard-coded
`rgeco`/`gcamp` defaults when a file uses names such as `green_signal`,
`red_signal`, or `iso_signal`.

Trace colours are also inferred from channel names.  For example, green/GCaMP
channels are drawn with a modern green palette, red/RGECO/tdTomato channels with
a muted red palette, and isosbestic/control channels with a neutral slate tone.


### Granger causality performance

Granger causality can be slow on full-resolution photometry traces because it
fits autoregressive models at multiple lags. The GUI defaults now use
interactive-safe settings: downsample to `target_dt`, cap the number of tested
lag steps, and cap the number of samples. For peak-aligned connectivity,
`granger_mode="mean_epoch"` tests the mean aligned waveform and is much faster
than `per_event`, which tests each event separately. Use scripts/offline runs for
large `per_event` analyses.
