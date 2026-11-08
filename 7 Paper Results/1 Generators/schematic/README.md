# Connectome learning overview (Figure 1)

Open `connectome-learning-overview.drawio` in diagrams.net to edit the schematic. Images are embedded in the diagram.

After restoring the Zenodo artifacts, export the saved diagram from the repository root:

```bash
python "7 Paper Results/1 Generators/schematic/build_schematic.py" --output "7 Paper Results/2 Outputs/schematic"
```

The command requires `rsvg-convert` (librsvg) with a working Fontconfig configuration and Arial or a compatible sans-serif font. It exports PDF, PNG and SVG files into a new directory. Use `--rebuild` to recreate diagram cells from the source layout and archived assets instead of the edited diagram.

The numerical snapshots, activity traces, image assets and attribution are in `5 Iris/5 Outputs/paper-schematic/`. The six panels show updates 0, 1, 6, 12, 18 and 24 of the same trained Iris model. Anatomical morphology is based on MaleCNS v1.0, distributed under CC BY 4.0. See the archived `sources.json` for attribution and source links.
