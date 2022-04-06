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
python -m hyper.run [OPTIONS]
```

Use the `--dont-generate-models` option if only the search is needed to be run (models already exists). 

Results will be stored as specified in the config file's `hyper_results_dir` param.

### Run figure generator
Generate figures containing the histogram and the plotted CDFs and PDFs.

```Bash
python -m cdf.create_figures.run
```
By default results will be in `cdf/create_figures/results`

### Run evaluation
Evaluate the best models on the weekly separated dataset. Set `weekly_root_dir` and `task_id_pattern` for a new run. On a new setup, change at least `eval_dir`, `sandbox_root`. Then:

```Bash
python -m evaluate.run
```

## Other

### Supervisor related
Configs are conventionally placed in `/etc/supervisor/conf.d`

An example config with conda env activated:
```
[program:generate_cdf]
 directory=<simbiota_tools_root>
 environment=PATH="<miniconda_path>/envs/dwf_client/bin:%(ENV_PATH)s"
 command=<miniconda_path>/envs/dwf_client/bin/python -m cdf.run
 stderr_logfile=/var/log/generate_cdf/supervisor_run.err.log
 stderr_logfile_maxbytes=10MB
 stderr_logfile_backups=0
```

#### Run

```Bash
sudo supervisorctl
reread
update
```

Then management goes like
```Bash
<start/stop/restart> generate_cdf
tail generate_cdf stderr
```

### Redis related
This works only on Linux. 

1. Setup a redis server
2. Setup rq-dashboard
3. Implement enqueueing into rq. Set `job_timeout` to at least 30m.