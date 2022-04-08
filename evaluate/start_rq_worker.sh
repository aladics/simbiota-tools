#!/bin/bash

source /home/aladics/miniconda3/bin/activate dwf_client
cd /home/dwf/simbiota/tools
rq worker simbiota
