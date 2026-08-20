# Data Card — Default Research Dataset

## Source

CarbonTwin-R v0.2 uses the **UCI Appliances Energy Prediction** dataset (UCI id 374) as its default real-data research substrate.

- Real experimental building measurements.
- 19,735 observations.
- 10-minute sampling over about 4.5 months.
- Appliance energy target plus lighting, nine indoor temperature/humidity zones and external weather variables.
- License: CC BY 4.0.
- DOI: 10.24432/C5VC8G.

The downloader records the source, license, file hash, byte count and download timestamp in `data/raw/uci_appliances/source_manifest.json`.

## Important distinction

The **base measurements are real**. Faults used for benchmark scoring are deliberately injected into the untouched chronological test period. Therefore:

- real measured context: yes;
- real measured field failures with labels: no;
- known fault onset/severity/healthy counterfactual for evaluation: yes, because the injected perturbation is controlled.

This is described as a **semi-synthetic fault benchmark on real data**, never as a measured industrial failure dataset.

## Excluded columns

`rv1` and `rv2` are intentionally random variables included with the UCI dataset. They are excluded from modelling and retained as a useful example of negative-control awareness.
