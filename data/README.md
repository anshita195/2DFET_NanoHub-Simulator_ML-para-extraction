
## Data layout
Each directory within [data/raw](data/raw) corresponds to a single transistor and 
contains three .csv files:
* variables.csv -- Values for each of the fitting parameters (see Table 1 in the
preprint for a description of the physical meaning of each parameter)
* IdVg_Vds=0.1.csv -- Id-Vgs curve at a fixed drain bias of 0.1 V
* IdVg_Vds=1.0.csv -- Id-Vgs curve at a fixed drain bias of 1 V

The latter two csv files are formatted as:
"x","y"
-5.95398,1.21043e-13
-5.84493,1.43213e-13
-5.63264,1.93936e-13
4.95395,1.48955e-10
6.55052,2.23486e-10
...


## Processing data
See the supplemental information in the preprint for details on how we process
our data. You can process this data into easy-to-work-with NumPy arrays by
running the script [process_data](./../demo/process_data.py).


