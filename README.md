# LSTM-Attention Runoff Simulation (1980–2020)

Climate-driven runoff modeling using an **LSTM + temporal attention** network with a standard **train–validation–simulation** split:
- Train: **1980–1996**
- Validation: **1997–2000**
- Simulation: **2001–2020**

Files:
- `Code.py` — main script (training + rolling simulation)
- `Data_Monthly_1980_2020.csv` — monthly climate feature dataset (1980–2020)

1.## Data format (CSV)
`Data_Monthly_1980_2020.csv` contains the following columns (A–J):
- A `data`
- B `snowmelt`
- C `pet`
- D `precipitation`
- E `surface_sensible_heat_flux`
- F `snowfall`
- G `temperature`
- H `u_component_of_wind`
- I `v_component_of_wind`
- J `runoff`

2.Edit Code.py to set file paths / configs as needed (this repo provides the core code only).

3.Notes on runoff data access
Due to local data-sharing policies, runoff observations and other restricted data are not publicly available.

