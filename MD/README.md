# Molecular dynamics simulation files

This directory contains the input files and system files used for the molecular dynamics simulations in EZPro-Fit.

## Contents

- `system.prmtop`: AMBER topology file
- `system.inpcrd`: AMBER initial coordinate/restart file
- `mdin/`: AMBER minimization, heating, equilibration, and production input files
- `tleap.in`: System preparation script
- `run_md.sh`: MD run script
- `MD_RMSD.png`: RMSD analysis figure

The simulations were performed using AMBER. The exact force-field, solvent-model, equilibration, production, and trajectory-analysis details are provided in the manuscript and reporting checklists.
