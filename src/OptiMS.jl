
include("helper_functions.jl")

##################################### import packages #####################################
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Random
using Base.Threads
using BlackBoxOptim
rng = MersenneTwister(1)
const K = 5


##################################### parse command-line arguments #####################################
args = parse_args()
query_data = args["--query_data"]
reference_data = args["--reference_data"]
output = args["--output"]
optimization_method = args["--optimization_method"]
metric = args["--metric"]
params_to_optimize = args["--params_to_optimize"]
spectrum_preprocessing_order = args["--spectrum_preprocessing_order"]
num_threads = parse(Int, args["--threads"])
n_grid_points = parse(Int, args["--n_grid_points"])
max_steps = parse(Int, args["--max_steps"])
LB_wf_mz = parse(Float64, args["--LB_wf_mz"])
UB_wf_mz = parse(Float64, args["--UB_wf_mz"])
LB_wf_intensity = parse(Float64, args["--LB_wf_intensity"])
UB_wf_intensity = parse(Float64, args["--UB_wf_intensity"])
LB_LET_thresh = parse(Float64, args["--LB_LET_thresh"])
UB_LET_thresh = parse(Float64, args["--UB_LET_thresh"])
wf_mz = parse(Float64, args["--wf_mz"])
wf_intensity = parse(Float64, args["--wf_intensity"])
LET_thresh = parse(Float64, args["--LET_thresh"])

if nthreads() < num_threads
    println("Warning: Julia was started with $(nthreads()) threads, cannot increase to $num_threads")
end

const PARAM_NAMES = ["wf_mz", "wf_int", "LET_thresh"]

function parse_params_to_optimize(s::AbstractString)
    s = strip(s)
    if s == "all"
        return collect(1:3)
    end
    parts = split(s, ',')
    parts = strip.(parts)
    name_to_idx = Dict("wf_mz" => 1, "wf_int" => 2, "wf_intensity" => 2, "LET_thresh" => 3)
    idxs = Int[]
    for p in parts
        haskey(name_to_idx, p) || error("Unknown param in --params_to_optimize: '$p'. Allowed: all, wf_mz, wf_int, LET_thresh (comma-separated).")
        push!(idxs, name_to_idx[p])
    end
    idxs = unique(idxs)
    sort!(idxs)
    return idxs
end

opt_idxs = parse_params_to_optimize(params_to_optimize)
fix_idxs = setdiff(collect(1:3), opt_idxs)
p_fixed_full = Float64[wf_mz, wf_intensity, LET_thresh]
LB_full = Float64[LB_wf_mz, LB_wf_intensity, LB_LET_thresh]
UB_full = Float64[UB_wf_mz, UB_wf_intensity, UB_LET_thresh]
bounds_free = [(LB_full[i], UB_full[i]) for i in opt_idxs]

function expand_params(p_free::AbstractVector{<:Real}, p_fixed_full::Vector{Float64}, opt_idxs::Vector{Int})
    p = copy(p_fixed_full)
    @assert length(p_free) == length(opt_idxs)
    for (j, idx) in enumerate(opt_idxs)
        p[idx] = Float64(p_free[j])
    end
    return p
end



##################################### import data #####################################
df_query_raw = CSV.read(query_data, DataFrame; delim='\t')
df_ref_raw = CSV.read(reference_data, DataFrame; delim='\t', pool=false, missingstring=["","NA","N/A"], stringtype=String)
q_ids_all = string.(df_query_raw[!, :id])
r_ids_all = string.(df_ref_raw[!, :id])
println("Note: There are $(length(setdiff(unique(q_ids_all),unique(r_ids_all)))) query spectra whose ground-truth ID is not contained in the reference library.")


##################################### run analysis #####################################
callback = oc -> begin
    push!(fitness_progress_history, (evals = BlackBoxOptim.num_func_evals(oc), fitness = BlackBoxOptim.best_fitness(oc), params = copy(BlackBoxOptim.best_candidate(oc))))
    false
end

Q0 = Matrix{Float64}(df_query_raw[:, 2:end])
R0 = Matrix{Float64}(df_ref_raw[:, 2:end])
m = size(Q0, 2)
@assert m == size(R0, 2) "Query and reference must have same number of columns"
mzs = collect(1.0:m)
folds = make_folds(size(Q0,1), K; rng=rng)
if metric == "accuracy"
    objective_full = objective_acc
else
    objective_full = objective_MRR
end

objective_free(p_free) = objective_full(expand_params(p_free, p_fixed_full, opt_idxs))


if optimization_method == "DE"
    fitness_progress_history = Vector{NamedTuple{(:evals, :fitness, :params_full), Tuple{Int, Float64, Vector{Float64}}}}()

    callback = oc -> begin
        p_free_best = BlackBoxOptim.best_candidate(oc)
        p_full_best = expand_params(p_free_best, p_fixed_full, opt_idxs)
        push!(fitness_progress_history, (
            evals = BlackBoxOptim.num_func_evals(oc),
            fitness = BlackBoxOptim.best_fitness(oc),
            params_full = p_full_best
        ))
        false
    end

    res = bboptimize(
        objective_free;
        SearchRange = bounds_free,
        NumDimensions = length(bounds_free),
        Method = :de_rand_1_bin,
        PopulationSize = 50,
        MaxSteps = max_steps,
        CallbackFunction = callback,
        CallbackInterval = 0.0,
        rng = rng
    )
elseif optimization_method == "grid"
    grid_full = [
        collect(range(LB_full[1], UB_full[1]; length=n_grid_points)),
        collect(range(LB_full[2], UB_full[2]; length=n_grid_points)),
        collect(range(LB_full[3], UB_full[3]; length=n_grid_points))
    ]

    grids_free = [grid_full[i] for i in opt_idxs]

    combos = collect(Iterators.product(grids_free...))
    n = length(combos)

    wf_mz_out   = Vector{Float64}(undef, n)
    wf_int_out  = Vector{Float64}(undef, n)
    LET_out     = Vector{Float64}(undef, n)
    fit_out     = Vector{Float64}(undef, n)

    Threads.@threads for i in 1:n
        p_free = collect(combos[i])
        p_full = expand_params(p_free, p_fixed_full, opt_idxs)
        fit = objective_full(p_full)
        wf_mz_out[i]  = p_full[1]
        wf_int_out[i] = p_full[2]
        LET_out[i]    = p_full[3]
        fit_out[i]    = fit
    end
    grid_results_tmp = DataFrame(wf_mz = wf_mz_out, wf_int = wf_int_out, LET_thresh = LET_out, fitness = fit_out)
    grid_results = sort!(grid_results_tmp, :fitness, rev=true)
elseif optimization_method == "none"
    S = get_scores(Q0, R0; order=spectrum_preprocessing_order, wf_mz=wf_mz, wf_int=wf_intensity, LET_thresh=LET_thresh, mzs=mzs)
    df_scores = DataFrame(S, r_ids_all)
    insertcols!(df_scores, 1, :query_id => q_ids_all)
end


if optimization_method == "DE"
    open(output, "w") do io
        header = ["step", "evals", "fitness", "wf_mz", "wf_int", "LET_thresh"]
        write(io, join(header, '\t') * "\n")
	for (step, rec) in enumerate(fitness_progress_history)
            row = String[
	        string(step-1),
                string(rec.evals),
                string(rec.fitness),
                string(rec.params_full[1]),
                string(rec.params_full[2]),
                string(rec.params_full[3])
            ]
            write(io, join(row, '\t') * "\n")
        end
    end
elseif optimization_method == "grid"
    CSV.write(output, grid_results; delim='\t')
elseif optimization_method == "none"
    CSV.write(output, df_scores; delim='\t')
end

