# Dataset source and attribution

Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.

- Dataset: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Original archive: https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
- Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The source contains 2,075,259 minute records for one household in Sceaux, France,
from December 2006 to November 2010. These CSVs are derived versions, not the raw
minute file. Changes: parse local-clock timestamps, average available minutes
within each hour, and forward-fill completely empty hours from earlier readings.
The two partial endpoint hours are retained. The result has 34,589 hourly rows.
No future observation is used to fill a gap.

`electricity_hourly.csv` contains demand plus audit columns. The expanded
`electricity_hourly_multifeature.csv` contains all seven electrical measurements
plus audit columns. It retains the exact same demand values. Submeter channels
are hourly means of the minute values, not hourly energy sums. Calendar encodings
and causal recent-demand summaries are constructed by the notebook, not stored
as additional observed electrical channels in the source.

The CSVs preserve source scale. Column names and target units are documented in
the notebook. Dataset audit details are in `data/source_metadata.json` and
`data/source_metadata_multifeature.json`. Retain this attribution when sharing
the derived datasets.

## Pinned checksums

| File | SHA-256 |
|---|---|
| UCI source ZIP | `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff` |
| Demand-only CSV | `afbfc86aad2fbc2b0466fab43ea7bf58d089b2d287e229dbb883eafaf8e8c758` |
| Expanded CSV | `9c3ccfeca53444126550867208723c4c1965a973fd290a34d2c9d2b8aba669a8` |
