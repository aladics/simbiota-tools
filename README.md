# Simbiota tools

## Setup:
- Start DWF anaconda environment
- Copy and rename `config_default.yaml` to `config.yaml` and set the values accordingly (hyper res and hyper cache dirs must be created manually)

## Usgage
Run commands from root.

### Run CDF generator

```Bash 
python -m cdf.run
```

### Run Hyperparam search

```Bash
python -m hyper.run
```

### Run figure generator
Generate figures containing the histogram and the plotted CDFs&PDFs.

```Bash
python -m cdf.create_figures.run
```
