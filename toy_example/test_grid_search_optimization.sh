#!/bin/bash

cd ${PWD}/..
echo -e "\nTuning parameters via exhaustive grid-search...\n"

julia --threads auto src/OptiMS.jl \
  --query_data toy_example/data/query_data.txt \
  --reference_data toy_example/data/reference_data.txt \
  --output toy_example/output_grid_tuning.txt \
  --optimization_method grid \
  --params_to_optimize all \
  --metric MRR \
  --n_grid_points 3 \
  --LB_wf_mz 0.0 \
  --UB_wf_mz 5.0 \
  --LB_wf_intensity 0.0 \
  --UB_wf_intensity 5.0 \
  --LB_LET_thresh 0.0 \
  --UB_LET_thresh 5.0

