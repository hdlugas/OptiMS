
using Base.Threads
using DataFrames
using Random

function print_help()
    println("""
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
    --LB_wf_mz <float> \
    --UB_wf_mz <float> \
    --LB_wf_intensity <float> \
    --UB_wf_intensity <float> \
    --LB_LET_thresh <float> \
    --UB_LET_thresh <float> \
    --wf_mz <float> \
    --wf_intensity <float> \
    --LET_thresh <float> \
    --threads <int> \
    --n_grid_points <int> \
    --max_steps <int>


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
""")
end


function parse_args()
    args = Dict{String, String}()
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--help" || ARGS[i] == "-h"
            print_help()
            exit(0)
        elseif startswith(ARGS[i], "--")
            if i == length(ARGS)
                error("Missing value for argument: ", ARGS[i])
            end
            args[ARGS[i]] = ARGS[i+1]
            i += 2
        else
            error("Unknown argument: ", ARGS[i])
        end
    end
    for req in ["--query_data", "--reference_data", "--output"]
        haskey(args, req) || error("Missing required argument: $req. Use --help for usage.")
    end
    get!(args, "--metric", "accuracy")
    get!(args, "--params_to_optimize", "all")
    get!(args, "--spectrum_preprocessing_order", "WL")
    get!(args, "--threads", "1")
    get!(args, "--n_grid_points", "2")
    get!(args, "--max_steps", "5")
    get!(args, "--LB_wf_mz", "0.0")
    get!(args, "--UB_wf_mz", "1.3")
    get!(args, "--LB_wf_intensity", "0.51")
    get!(args, "--UB_wf_intensity", "1.0")
    get!(args, "--LB_LET_thresh", "0.0")
    get!(args, "--UB_LET_thresh", "3.0")
    get!(args, "--wf_mz", "0.0")
    get!(args, "--wf_intensity", "1.0")
    get!(args, "--LET_thresh", "0.0")
    if !(args["--metric"] in ["accuracy","MRR"])
        println("Warning: metric must be either 'accuracy' or 'MRR'.")
    end
    if !(args["--spectrum_preprocessing_order"] in ["","L","W","WL","LW"])
        println("Warning: spectrum_preprocessing_order must be either '', 'L', 'W', 'WL', or 'LW'.")
    end
    if !(args["--params_to_optimize"] in ["all","wf_mz","wf_int","LET_thresh","wf_mz,wf_int","wf_mz,LET_thresh","wf_int,LET_thresh"])
        println("Error: invalid params_to_optimize parameter. Run <julia OptiMS.jl --help> for usage instructions")
    end
    if !(args["--optimization_method"] in ["DE","grid","none"])
        println("Warning: optimization_method must be either 'DE', 'grid', or 'none'.")
    end
    return args
end


function make_folds(n::Int, K::Int; rng::AbstractRNG)
    idx = collect(1:n)
    shuffle!(rng, idx)
    folds = [Int[] for _ in 1:K]
    for (t, i) in enumerate(idx)
        push!(folds[mod1(t, K)], i)
    end
    return folds
end



function wf_transformation(X::AbstractMatrix, wf_mz::Real, wf_int::Real; mzs::AbstractVector)
    wrow = reshape(mzs .^ wf_mz, 1, :)
    return (X .^ wf_int) .* wrow
end


function LE_transformation(X::AbstractMatrix, LET_thresh::Real)
    Xf = Array{Float64}(X)
    rs = sum(Xf, dims=2)
    P  = similar(Xf); P .= 0.0
    nz = vec(rs) .> 0.0
    P[nz, :] .= Xf[nz, :] ./ rs[nz, :]
    T = zeros(size(P))
    idx = P .> 0.0
    @inbounds T[idx] .= P[idx] .* log.(P[idx])
    Sv = vec(-sum(T, dims=2))  # entropy per row
    lt_mask = (Sv .> 0.0) .& (Sv .< LET_thresh)
    if any(lt_mask)
        w = (1 .+ Sv) ./ (1 .+ LET_thresh)
        out = copy(P)
        @threads for i in eachindex(Sv)
            if lt_mask[i]
                @inbounds out[i, :] .= P[i, :].^w[i]
            end
        end
        return out
    else
        return P
    end
end



function row_l2_normalize!(X::AbstractMatrix)
    norms = sqrt.(sum(abs2, X; dims=2))
    nz = vec(norms) .> 0.0
    X[nz, :] .= X[nz, :] ./ norms[nz, :]
    return X
end


function get_acc(Q_mat::AbstractMatrix, R_mat::AbstractMatrix,
                         q_ids_all::AbstractVector{<:AbstractString},
                         r_ids_all::AbstractVector{<:AbstractString})
    Qn = copy(Q_mat); Rn = copy(R_mat)
    row_l2_normalize!(Qn); row_l2_normalize!(Rn)
    S = Qn * Rn'
    preds = Vector{String}(undef, size(S, 1))
    hits  = falses(size(S, 1))
    @threads for i in 1:size(S, 1)
        srow = @view S[i, :]
        _, j = findmax(srow)
        preds[i] = String(r_ids_all[j])
        hits[i] = (String(q_ids_all[i]) == preds[i])
    end
    acc = count(hits) / length(hits)
    return acc
end


function get_MRR(Q_mat::AbstractMatrix, R_mat::AbstractMatrix,
                 q_ids_all::AbstractVector{<:AbstractString},
                 r_ids_all::AbstractVector{<:AbstractString})
    Qn = copy(Q_mat); Rn = copy(R_mat)
    row_l2_normalize!(Qn); row_l2_normalize!(Rn)
    S = Qn * Rn'
    ref_index = Dict{String, Int}(String(r_ids_all[j]) => j for j in eachindex(r_ids_all))
    rr = Vector{Float64}(undef, size(S, 1))
    @threads for i in 1:size(S, 1)
        srow = @view S[i, :]
        _, jmax = findmax(srow)
        qid = String(q_ids_all[i])
        jtrue = ref_index[qid]
        ord = sortperm(srow; rev=true)
        rnk = findfirst(==(jtrue), ord)
        rr[i] = 1.0 / rnk
    end
    MRR = sum(rr) / length(rr)
    return MRR
end


function get_scores(Q0, R0; order=spectrum_preprocessing_order, wf_mz=wf_mz, wf_int=wf_int, LET_thresh=LET_thresh, mzs=mzs)
    Qn = copy(Q0); Rn = copy(R0)
    Qp, Rp = apply_pipeline(Qn, Rn; order = order, wf_mz = wf_mz, wf_int = wf_int, LET_thresh = LET_thresh, mzs = mzs)
    row_l2_normalize!(Qp); row_l2_normalize!(Rp)
    S = Qp * Rp'
    return S
end


function apply_pipeline(Q::AbstractMatrix, R::AbstractMatrix; order::AbstractString, wf_mz::Real, wf_int::Real, LET_thresh::Real, mzs::AbstractVector)
    Q_mat = Array{Float64}(Q)
    R_mat = Array{Float64}(R)
    for c in order
        if c == 'W'
            Q_mat = wf_transformation(Q_mat, wf_mz, wf_int; mzs=mzs)
            R_mat = wf_transformation(R_mat, wf_mz, wf_int; mzs=mzs)
        elseif c == 'L'
            Q_mat = LE_transformation(Q_mat, LET_thresh)
            R_mat = LE_transformation(R_mat, LET_thresh)
        else
            error("Unknown transform code '$c'. Use only 'W','L'.")
        end
    end
    return Q_mat, R_mat
end


function objective_acc(x)
    wf_mz, wf_int, LET_thresh = x
    min_acc = 99999
    for k in 1:K
        val_idx = folds[k]
        Q0_val = @view Q0[val_idx, :]
        q_ids_val = q_ids_all[val_idx]
        Qp_val, Rp = apply_pipeline(Q0_val, R0; order = spectrum_preprocessing_order, wf_mz = wf_mz, wf_int = wf_int, LET_thresh = LET_thresh, mzs = mzs)
        acc_k = get_acc(Qp_val, Rp, q_ids_val, r_ids_all)
        min_acc = min(min_acc, acc_k)
        if min_acc == 0.0
            break
        end
    end
    return 1.0 - min_acc
end


function objective_MRR(x)
    wf_mz, wf_int, LET_thresh = x
    min_MRR = Inf
    for k in 1:K
        val_idx = folds[k]
        Q0_val = @view Q0[val_idx, :]
        q_ids_val = q_ids_all[val_idx]
        Qp_val, Rp = apply_pipeline(Q0_val, R0; order = spectrum_preprocessing_order, wf_mz = wf_mz, wf_int = wf_int, LET_thresh = LET_thresh, mzs = mzs)
        MRR_k = get_MRR(Qp_val, Rp, q_ids_val, r_ids_all)
        min_MRR = min(min_MRR, MRR_k)
        if min_MRR == 0.0
            break
        end
    end
    return 1.0 - min_MRR
end


