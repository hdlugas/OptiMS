#!/bin/bash

cd ${PWD}/..
echo -e "\nRunning compound identification...\n"

julia --threads auto src/OptiMS.jl \
  --query_data toy_example/data/query_data.txt \
  --reference_data toy_example/data/reference_data.txt \
  --output toy_example/output_similarity_scores.txt \
  --optimization_method none \
  --wf_mz 0.5 \
  --wf_intensity 1.5 \
  --LET_thresh 3.0

