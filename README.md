# GOPS Solver

Nash-equilibrium tooling for Goofspiel (GOPS), focused on **maximizing winrate**.

Game variant used here:
- Tied bids are worth 0 points for that prize.
- No carry-over on ties.

Primary objective used for strategy is win/loss:
- Win = `+1`
- Draw = `0`
- Loss = `-1`

`points` objective is kept for transfer/validation experiments, not as the main play objective.

## Main Interactive Interface

The primary interactive interface is the Streamlit viewer.

- Hosted app: https://goofspiel-ne-viewer.streamlit.app
- Local launch:

```bash
streamlit run ai/evc_viewer.py
```

## Repository Map

- `solver/`: C++ solver and `.evc` cache generation.
- `ai/`: Python analysis/play tooling (`play.py`, `exploitability.py`, `find.py`, viewer app).
- `reports/`: generated caches and metadata.
- `tests/`: Python tests (including C++ parity checks).
- `docs/math.md`: derivation-style writeup.
- `ai/theory.md`: strategy notes and intuition.

## Quickstart

### 1) Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Build `cpp_solver`

Windows (vcpkg preset):

```powershell
Set-Location solver
cmake --preset vcpkg
cmake --build --preset vcpkg
Set-Location ..
```

### 3) Generate a cache

Windows:

```powershell
.\solver\build\vcpkg\Release\cpp_solver.exe --n 8 --cache-out reports/full8.evc
```

## Common Workflows

Play against NE bot:

```bash
python ai/play.py reports/full8.evc 8 0
```

Run strategy-vs-strategy simulation:

```bash
python ai/play.py reports/full8.evc --auto --n 8 --count 1000 --sa evc-ne --sb random --seed 1
```

Measure exploitability:

```bash
python ai/exploitability.py --policy evc-ne --cache reports/full8.evc --n 8
```

Mine NE support trends:

```bash
python ai/find.py --cache reports/full8.evc --sample-size 2000 --top-k 8
```

Launch cache viewer UI (primary interactive interface):

```bash
streamlit run ai/evc_viewer.py
```

Hosted app: https://goofspiel-ne-viewer.streamlit.app

## Documentation

- Friendly usage starts here in `README.md`.
- Math/notation writeup: `docs/math.md`.
- Additional strategy thoughts: `ai/theory.md`.
