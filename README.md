# Cardiomyocyte Morphodynamic Simulator

Streamlit application that synthesises and animates cardiomyocyte morphological maturation and subsequent degeneration across an eight-day culture window, driven by a parametric stage model.

> **Scope note.** This repository contains a generative visualisation tool. Every frame is drawn procedurally from hand-specified stage descriptors: no microscopy acquisition is read, no optical-flow field is estimated, and no quantitative measurement is produced. Despite the repository name, there is currently no cell-motion estimation code here. The application is a didactic and figure-generation aid, not an image-analysis pipeline.

## Architecture

![Processing architecture of the cardiomyocyte morphodynamic simulator](docs/architecture.svg)

*Figure 1. Vector schematic of the processing chain, from the discrete stage descriptor set through temporal interpolation, procedural rasterisation and sequence encoding to the interactive front end.*

## Stage parameter model

Eight discrete states are defined over the culture window. Intermediate time points are obtained by interpolation rather than by simulation of an underlying biophysical model.

| Day | Stage | Elongation | Alignment | Sarcomere order | Beat amplitude | Beat synchrony | Debris |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Immature | 1.0 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 2 | Initial beating | 1.2 | 0.2 | 0.2 | 0.3 | 0.2 | 0.1 |
| 3 | Mean beating begins | 1.5 | 0.4 | 0.4 | 0.5 | 0.4 | 0.2 |
| 4 | Stronger contractions | 1.8 | 0.6 | 0.6 | 0.7 | 0.6 | 0.2 |
| 5 | Moderate synchronisation | 2.0 | 0.8 | 0.8 | 0.85 | 0.8 | 0.3 |
| 6 | Peak contractile activity | 2.2 | 0.9 | 0.9 | 1.0 | 0.9 | 0.4 |
| 7 | Fragmentation onset | 1.6 | 0.5 | 0.5 | 0.6 | 0.5 | 0.7 |
| 8 | Significant damage | 1.2 | 0.2 | 0.1 | 0.2 | 0.1 | 0.9 |

Additional descriptors held per state: intercellular coupling, nucleus-to-cell size ratio, cell count, cluster cohesion and an RGB base colour.

## Method

**Temporal interpolation.** For a requested day `t`, scalar descriptors are interpolated piecewise-linearly between the bracketing integer states; the base colour is interpolated independently on each RGB channel. Categorical fields such as the stage label and shape class are inherited from the lower bracket. Results are memoised with `st.cache_data`.

**Frame synthesis.** Two to three cluster centroids are drawn uniformly inside the canvas. Cells are placed by sampling a polar offset within a cluster radius that scales with cluster cohesion. Each cell is rendered as an ellipse whose aspect ratio follows the elongation descriptor and whose size is modulated by a beat gain of the form `1 + A sin(pi phi)`, where `phi` is the phase at the current time point. Mature states receive a horizontal sarcomere raster; fragmenting states are decomposed into sub-ellipse debris. A nucleus ellipse is drawn for approximately 80 percent of cells, and a stochastic debris field is superimposed with density proportional to the debris descriptor.

**Sequence encoding.** Twelve to sixteen keyframes are sampled uniformly across the selected day range and written by the Pillow GIF writer with an infinite loop and a user-selected inter-frame delay of 125, 250 or 500 ms.

## Repository contents

| Path | Role |
| --- | --- |
| `complete-cardiomyocyte-app.py` | Stage model, interpolation, renderer, encoder and Streamlit front end |
| `requirements.txt` | Pinned Python environment |
| `packages.txt` | System packages for Streamlit Community Cloud |
| `docs/architecture.svg` | Vector source for Figure 1 |

## Running locally

```bash
pip install -r requirements.txt
streamlit run complete-cardiomyocyte-app.py
```

## Interface

Controls: animation speed (slow, medium, fast), day range (all days, early, middle, late) and a preview-day slider quantised to 0.5-day steps. Outputs: a cached single-frame preview, the generated sequence and a GIF download.

## Known limitations

- The repository name refers to cell-motion analysis, but no optical-flow or tracking code is present.
- Rendered frames are synthetic. They are neither derived from nor validated against acquired microscopy.
- `requirements.txt` is a full environment freeze. Several pinned packages, including `torch`, `keras` and `opencv-python-headless`, are not imported by the application.
- The random number generator is not seeded, so repeated runs produce different frames.

## Roadmap

1. Add a dense optical-flow module (Farneback or TV-L1) operating on acquired live-cell sequences, producing per-frame contraction amplitude and velocity fields.
2. Calibrate the stage descriptors against measured contraction metrics rather than hand-tuned values.
3. Seed the generator so that published figures are reproducible.
4. Reduce `requirements.txt` to the packages the application actually imports.

## Related repositories

- `Cardiac-Cell-Development-Animation-Day-1-8-` - closely related animation of the same eight-day window.
- `Deep-Spatiotemporal-Modelling-of-Cardiomyocyte-Ageing-Dysfunction` - optical-flow driven analysis of cardiomyocyte ageing.
- `Cardiomyocyte-Ageing-Nucleus-Data-Analysis` - nuclear morphology and chromatin biomarkers of ageing.

## Licence

No licence has been declared for this repository yet.

## Author

Md Abu Sufian - ORCID [0009-0007-3503-6942](https://orcid.org/0009-0007-3503-6942)
