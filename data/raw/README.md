# BCI Competition IV 2a Dataset

This directory is where local raw EEG files should be placed. Raw dataset files are ignored by Git.

## Dataset Summary

| Property | Value |
|---|---|
| Dataset | BCI Competition IV Dataset 2a |
| Subjects | 9 |
| Sessions | Training and evaluation |
| EEG channels | 22 |
| EOG channels | 3, not used by default |
| Sampling rate | 250 Hz |
| Trial window | 3 seconds, 750 samples |
| Classes | Left hand, right hand, feet, tongue |
| File format | MATLAB `.mat` |

## Download Sources

- Primary page: http://www.bbci.de/competition/iv/
- BNCI Horizon 2020 mirror: https://bnci-horizon-2020.eu/database/data-sets/001-2014/

Check the source site for the latest terms of use before redistributing data.

## Expected Files

Place files in this directory:

```text
data/raw/
|-- A01T.mat
|-- A01E.mat
|-- A02T.mat
|-- A02E.mat
|-- ...
|-- A09T.mat
`-- A09E.mat
```

The default loaders use `A0*T.mat` training-session files. Evaluation files can be added for extended testing.

## Download Script

```python
import os
import urllib.request

base_url = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"
os.makedirs("data/raw", exist_ok=True)

for subject in range(1, 10):
    for session in ["T", "E"]:
        filename = f"A0{subject}{session}.mat"
        output_path = os.path.join("data", "raw", filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, output_path)
```

## Citation

```text
Brunner C, Leeb R, Mueller-Putz G, Schloegl A, Pfurtscheller G (2008).
BCI Competition 2008 - Graz data set A.
Institute for Knowledge Discovery, Graz University of Technology.
```

## Synthetic Fallback

If no `.mat` files are present, the code generates deterministic synthetic EEG for tests and smoke runs. Synthetic fallback output is not a scientific benchmark.
