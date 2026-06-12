## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

ffmpeg is also needed to save `.mp4` files. Install with:

```bash
# Mac
brew install ffmpeg

# or with conda
conda install -c conda-forge ffmpeg
```

---

## Run

```bash
python3 main2.py
```

Results are saved to the `results/` folder. Each subfolder has an `info.txt` explaining what the files are.

---

## Results structure

```
results/
├── rk2/            — RK2 vorticity simulation
├── q11_nu1000/     — Strouhal analysis, nu=1000
├── q11_nu5000/     — Strouhal analysis, nu=5000
└── island/         — Cabo Verde island simulation (RK4)
```
