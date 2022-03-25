# Simbiota tools

## Setup:
- Start DWF anaconda environment
- Copy and rename `config_default.yaml` to `config.yaml` and set the values accordingly (hyper res and hyper cache dirs must be created manually)
- If previous runs were ran:

```Bash 
python -m common.reset_all
```

## Usgage
Run commands from root.

### Run CDF generator

```Bash 
python -m cdf.run
```

Results will be stored as specified in the config file's `cdf_results_dir` param.

### Run Hyperparam search
Run cdf generator first. Then:

```Bash
python -m hyper.run
```

Results will be stored as specified in the config file's `hyper_results_dir` param.

### Run figure generator
Generate figures containing the histogram and the plotted CDFs and PDFs.

```Bash
python -m cdf.create_figures.run
```
By default results will be in `cdf/create_figures/results`