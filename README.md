# wrfchemigen

`wrfchemigen` is a tool for generating anthropogenic pollutant emissions for WRF-Chem simulations, specifically focusing on the Japan region using J-STREAM data.

**Status:** Under development

## Overview

This project provides scripts to process and convert anthropogenic emission datasets into a format compatible with WRF-Chem. The current implementation focuses on J-STREAM (Japan-Simplified Transport and Emission Model) inputs, based on the work of Dr. Satoru Chatani (NIES).

The tool handles:
- Mapping of emission species (e.g., from CB6/SAPRC to Mozart-Mosaic).
- Spatial regridding using `xesmf` and `geopandas`.
- Temporal allocation of emissions.
- Parallel processing using MPI for efficiency.

## Repository Structure

- `jstream/wrfchem_generator/`: Contains the core generation scripts.
  - `wrfchem_generator_jstream_parallel.py`: The main Python processing script using `mpi4py`.
  - `job_wrfchem_generate.bash`: A sample submission script for high-performance computing (HPC) environments.
  - `README.md`: Specific details regarding the J-STREAM generator.
- `.github/`: GitHub configuration and issue templates.

## Requirements

The following Python libraries are primarily used:
- `numpy`, `pandas`, `xarray`
- `geopandas`, `shapely`, `pyproj`
- `xesmf`, `jismesh`
- `mpi4py`, `f90nml`

## Usage

Configuration is typically handled via a namelist and paths defined within the `wrfchem_generator_jstream_parallel.py` script. Execution in parallel can be done via `mpirun`:

```bash
mpirun -n <num_procs> python wrfchem_generator_jstream_parallel.py
```

## Credits

- **Author:** Alvin C.G. Varquez (Science Tokyo)
- **Email:** varquez.a.aa@m.titech.ac.jp
- **J-STREAM Inputs:** Dr. Satoru Chatani (NIES)

## License

See the [LICENSE](LICENSE) file for details.
