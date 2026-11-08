# 1. Environment

## 1. Install

Use Python 3.12 in a Linux/Bash environment. On Windows, run the commands in WSL2; the launchers require POSIX file locking. Saved-model inference and paper reconstruction support CPU execution. CUDA is optional for those workflows.

From the repository root, create an environment and install the recorded direct dependency versions:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r "1 Environment/vision-validated.txt" -e '.[vision,ml,xai]'
./run.sh --help
```

The `vision` extra includes image-encoder dependencies. The `xai` extra provides SHAP, LIME and image segmentation. Figure 1 export requires `rsvg-convert` (librsvg). The `ml` extra installs the additional traditional-classifier dependencies. For exact ML model replay, also install the versions in [`requirements.txt`](<../6 Ki67/2 Source/ml_baselines/requirements.txt>). The additional Iris classifier campaign records a separate [requirements file](<../5 Iris/2 Execution/topology_controls/requirements.txt>). Use a separate environment with those versions for exact Iris model replay. Numerical table reconstruction reads saved results and does not require loading classifier weights. Repeat the editable installation if the checkout is moved.

`./run.sh` launches connectome and Iris commands; `"6 Ki67/run.sh"` launches Ki-67 commands.

Next, [restore the Zenodo artifacts](<../4 Trained Models/README.md>) before using saved models or reconstructing paper results.

## 2. Dependencies and hardware

The experiments used Python 3.12.13 and PyTorch `2.2.2+cu121`. [`validated.txt`](validated.txt) records the direct Iris dependencies; [`vision-validated.txt`](vision-validated.txt) adds image-extraction dependencies. They do not lock transitive packages. For an Iris-only installation, use `validated.txt` and `-e .` instead.

Install a PyTorch CPU or CUDA wheel appropriate to the machine before the command above if its default wheel is unsuitable. GPU execution also requires a compatible NVIDIA driver. See [computational resources](<../8 Documentation/full_graph_compute.md>) and the [Ki-67 guide](<../6 Ki67/README.md>) for GPU selection and training requirements.

## 3. Interpreter overrides

Both launchers use the activated environment by default. `FLYBRAIN_PYTHON` can select an interpreter explicitly; otherwise `FLYBRAIN_ENV` selects the environment's `bin/python`. For example, `FLYBRAIN_ENV=.venv ./run.sh --help` uses the environment created above.
