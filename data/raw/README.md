# BCI Competition IV 2a Dataset

## Overview
This directory stores the raw `.mat` files for the BCI Competition IV Dataset 2a.

## Download Instructions

### Official Source
The dataset is publicly available from the BCI Competition IV website:

**Primary URL:** http://www.bbci.de/competition/iv/

**Mirror (BNCI Horizon 2020):** https://bnci-horizon-2020.eu/database/data-sets/001-2014/

### Files Required
Place all 18 `.mat` files (9 training + 9 evaluation) in this folder:

```
data/raw/
├── A01T.mat    ← Subject 1, Training session
├── A01E.mat    ← Subject 1, Evaluation session
├── A02T.mat    ← Subject 2, Training session
├── A02E.mat    ← Subject 2, Evaluation session
├── A03T.mat
├── A03E.mat
├── A04T.mat
├── A04E.mat
├── A05T.mat
├── A05E.mat
├── A06T.mat
├── A06E.mat
├── A07T.mat
├── A07E.mat
├── A08T.mat
├── A08E.mat
├── A09T.mat
└── A09E.mat
```

> **Note:** The code uses only `A0*T.mat` (training) files by default.
> Evaluation files (`A0*E.mat`) can be added as extra test subjects.

### Download Script (Colab)
```python
import os, urllib.request

os.makedirs("data/raw", exist_ok=True)
BASE = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"

for i in range(1, 10):
    for session in ["T", "E"]:
        fname = f"A0{i}{session}.mat"
        dest = f"data/raw/{fname}"
        if not os.path.exists(dest):
            print(f"Downloading {fname} ...", end=" ", flush=True)
            urllib.request.urlretrieve(BASE + fname, dest)
            print("done")
```

## Dataset Statistics

| Property            | Value                     |
|---------------------|---------------------------|
| Subjects            | 9                         |
| Sessions per subject| 2 (training + evaluation) |
| EEG channels        | 22                        |
| EOG channels        | 3 (not used)              |
| Sampling rate       | 250 Hz                    |
| Trial duration      | 3 seconds (750 samples)   |
| Classes             | 4 (left hand, right hand, feet, tongue) |
| Trials per class    | ~72 per session           |
| Total trials        | ~288 per subject (T session) |
| File format         | MATLAB .mat               |
| License             | Public domain (open access) |

## Motor Imagery Classes

| Class | Label | Description              |
|-------|-------|--------------------------|
| 1     | Left  | Left hand motor imagery  |
| 2     | Right | Right hand motor imagery |
| 3     | Feet  | Both feet motor imagery  |
| 4     | Tongue| Tongue motor imagery     |

## Privacy & Ethics
- This dataset contains NO real patient data
- Data was collected in a controlled laboratory setting
- All recordings are anonymized
- Distribution is unrestricted for research

## Citation
```
Brunner C, Leeb R, Müller-Putz G, Schlögl A, Pfurtscheller G (2008).
BCI Competition 2008 – Graz data set A.
Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology.
```

## Synthetic Fallback
If no `.mat` files are present, the code automatically generates synthetic EEG data
for testing and development. This allows running the full pipeline without downloading
the dataset. Synthetic data will not produce meaningful accuracy results.
