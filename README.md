# OptiMS
OptiMS is a julia-based command-line tool for tuning parameters involved in preprocessing mass spectrometry data. The three main functionalities of OptiMS are (i) identifying optimal parameters via differential evolution, (ii) identifying optimal parameters via exhaustive grid-based search, and (iii) simply running compound identification and recording similarity scores.


## Table of Contents
- [1. Install dependencies](#install-dependencies)
- [2. Parameter descriptions](#param-descriptions)
- [3. Functionality](#functionality)
   - [3.1 Optimize parameters via differential evolution](#DE)
   - [3.2 Optimize parameters via exhausive grid search](#grid-search)
   - [3.3 Run compound identification](#run-compound-identification)


<a name="install-dependencies"></a>
## 1. Install dependencies
The Julia packages required to run OptiMS are BlackBoxOptim, CSV, DataFrames, LinearAlgebra, Random, and Statistics. These dependencies can be installed with the Julia command:

```
using Pkg; Pkg.add(["BlackBoxOptim", "CSV", "DataFrames", "LinearAlgebra","Random", "Statistics"])
```

<a name="param-descriptions"></a>
# 2. Parameter descriptions
The following two spectrum preprocessing transformations are offered:

-   Weight Factor Transformation: Given a pair of user-defined weight
    factor parameters $(\text{a,b})$ and spectrum $I$ with m/z values
    $(m_{1},m_{2},...,m_{n})$ and intensities $(x_{1},x_{2},...,x_{n})$,
    the transformed spectrum $I^{\star}$ has the same m/z values as $I$
    and has intensities given by
    $I^{\star}:=(m_{1}^{\text{a}}\cdot x_{1}^{\text{b}},m_{2}^{\text{a}}\cdot x_{2}^{\text{b}},...,m_{n}^{\text{a}}\cdot x_{n}^{\text{b}})$.

-   Low-Entropy Transformation: Given a user-defined low-entropy
    threshold parameter $T$ and spectrum $I$ with intensities
    $(x_{1},x_{2},...,x_{n})$, $\sum_{i=1}^nx_i = 1$, and Shannon
    entropy $H_{Shannon}(I)=-\sum_{i=1}^{n}x_{i}\cdot ln(x_{i})$, the
    transformed spectrum intensities
    $I^{\star}=(x_{1}^{\star},x_{2}^{\star},...,x_{n}^{\star})$ are such
    that, for all $i\in\{1,2,...,n\}$, $x_{i}^{\star}=x_{i}$ if
    $H_{Shannon}(I)\geq T$ and
    $x_{i}^{\star}=x_{i}^{\frac{1+H_{Shannon}(I)}{1+T}}$ if
    $H_{Shannon}(I)<T$.

Thus, there are two weight factor and one low-entropy threshold parameter one can tweak.

Given a pair of processed spectra intensities
$I=(a_{1},a_{2},...,a_{n}), J=(b_{1},b_{2},...,b_{n})\in\mathbb{R}^{n}$
with $0\leq a_{i},b_{i}\leq 1$ for all $i\in\{1,2,...,n\}$ and
$\sum_{i=1}^{n}a_{i}=\sum_{i=1}^{n}b_{i}=1$, PyCompound provides
functionality for computing the following similarity measures:

-   Cosine Similarity Measure:

```math
S_{Cosine}(I,J)=\frac{I\circ J}{|I|_{2}\cdot |J|_{2}}
```
where multiplication in the numerator refers to the dot product $I\circ J=a_{1}b_{1}+a_{2}b_{2}+...+a_{n}b_{n}$ of $I$ and $J$ and multiplication in the denominator refers to multiplication of the $L^{2}$-norms of $I$ and $J$, $\vert I\vert_{2}=\sqrt{a_{1}^{2}+a_{2}^{2}+...+a_{n}^{2}}, \vert J\vert_{2}=\sqrt{b_{1}^{2}+b_{2}^{2}+...+b_{n}^{2}}$.


<a name="functionality"></a>
## 3. Functionality

OptiMS has three main capabilities:
1. Plotting a query spectrum vs. a reference spectrum before and after preprocessing transformations.
2. Running spectral library matching to identify compounds based on their mass spectrometry data
3. Tuning parameters to maximize accuracy given a query dataset with known compuond IDs (e.g. from targeted metabolomics experiments).

Scripts which run toy examples illustrating each of these three methods are provided. These toy examples can be run by navigating to the necessary directory and executing the scripts:
```
cd toy_example
./test_DE_optimization.sh
./test_grid_optimization.sh
./test_run_compound_identification.sh
```

To view the OptiMS usage instruction, one can run the following from the command-line (once the necessary Julia dependencies are installed):
```
julia src/OptiMS.jl --help
```

The complete usage instructions for OptiMS are:
```
Help Message for OptiMS.jl

Usage:
  julia OptiMS.jl \
    --query_data <string> \
    --reference_data <string> \
    --output <string> \
    --optimization_method <string> \
    --metric <string> \
    --params_to_optimize <string> \
    --spectrum_preprocessing_order <string> \
    --n_grid_points <int> \
    --max_steps <int> \
    --LB_wf_mz <float> \
    --UB_wf_mz <float> \
    --LB_wf_intensity <float> \
    --UB_wf_intensity <float> \
    --LB_LET_thresh <float> \
    --UB_LET_thresh <float> \
    --wf_mz <float> \
    --wf_intensity <float> \
    --LET_thresh <float> \
    --threads <int>


Arguments:
  --query_data                    Path to input TXT file of query dataset (required).
  --reference_data                Path to input TXT file of reference dataset (required).
  --output                        Path to output TXT file (required).
  --optimization_method           Optimization approach (optional, options=[DE,grid,none], default=DE).
  --metric                        Quantity to maximize in the objective function (optional, options=[accuracy,MRR], default=accuracy).
  --params_to_optimize            String denoting the parameters to optimize (optional, options='all','wf_mz','wf_int','LET_thresh','wf_mz,wf_int','wf_mz,LET_thresh','wf_int,LET_thresh', default='all').
  --spectrum_preprocessing_order  String denoting the order of spectrum preprocessing transformations; format must be a string with 0-2 characters of either L (low-entropy trannsformation) and/or W (weight-factor-transformation) (optional, default: 'WL').
  --threads                       Number of threads to use (optional, default=1).
  --n_grid_points                 Number of grid points to use for each parameter; only applicable for grid-based optimization (optional, default=2).
  --max_steps                     Maximum number of iterations allowed in differential evolution optimization; only applicable for DE optimization_method (optional, default=5).
  --LB_wf_mz                      Float denoting the lower bound of the mass/charge weight factor parameter (optional, default=0.0).
  --UB_wf_mz                      Float denoting the upper bound of the mass/charge weight factor parameter (optional, default=1.3).
  --LB_wf_intensity               Float denoting the lower bound of the intensity weight factor parameter (optional, default=0.51).
  --UB_wf_intensity               Float denoting the upper bound of the intensity weight factor parameter (optional, default=1.0).
  --LB_LET_thresh                 Float denoting the lower bound of the low-entropy threshold parameter (optional, default=0.0).
  --UB_LET_thresh                 Float denoting the upper bound of the low-entropy threshold parameter (optional, default=3.0).
  --wf_mz                         Float denoting the mass/charge weight factor parameter; only applicable for optimization_method = none (optional, default=0.0).
  --wf_intensity                  Float denoting the intensity weight factor parameter; only applicable for optimization_method = none (optional, default=1.0).
  --LET_thresh                    Float denoting the low-entropy threshold parameter; only applicable for optimization_method = none (optional, default=0.0).
  --help                          Show this help message.
```

<a name="DE"></a>
### 3.1 Optimize parameters via differential evolution
To identify optimal parameters to maximize the metric (e.g. accuracy in this case) using differential evolution optimization with user-specified parameter bounds and maximum number of steps, one can run:
```
julia --threads auto src/OptiMS.jl \
  --query_data toy_example/data/query_data.txt \
  --reference_data toy_example/data/reference_data.txt \
  --output toy_example/output_DE_tuning.txt \
  --optimization_method DE \
  --params_to_optimize all \
  --metric accuracy \
  --max_steps 20 \
  --LB_wf_mz 0.0 \
  --UB_wf_mz 5.0 \
  --LB_wf_intensity 0.0 \
  --UB_wf_intensity 5.0 \
  --LB_LET_thresh 0.0 \
  --UB_LET_thresh 5.0
```

<a name="grid-search"></a>
### 3.2 Optimize parameters via exhaustive grid search
To record the metric (e.g. MRR in this case) for each combination of parameters in a user-specified grid of parameters with user-specified parameter bounds, one can run:
```
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
```

<a name="run-compound-identification"></a>
### 3.3 Run compound identification
To simply run compound identification and record all similarity scores with user-specified parameters, one can run:
```
julia --threads auto src/OptiMS.jl \
  --query_data toy_example/data/query_data.txt \
  --reference_data toy_example/data/reference_data.txt \
  --output toy_example/output_similarity_scores.txt \
  --optimization_method none \
  --wf_mz 0.5 \
  --wf_intensity 1.5 \
  --LET_thresh 3.0
```

