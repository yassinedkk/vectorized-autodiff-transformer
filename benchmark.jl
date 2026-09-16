##############################
# bench_hvp.jl  
##############################
using LinearAlgebra
using Random
using Statistics
using Printf

include("flatten.jl")
include("models.jl")               
include("forward.jl")               
include("reverse_vectorized.jl")    

const RNG = MersenneTwister(42)

# -------------------------
# Utilitaires
# -------------------------
flatten_vec(x) = reduce(vcat, vec.(x.components))
nparams(w::Flatten) = length(w)

# minuterie simple (min des temps sur `reps` exécutions)
function time_min(f; reps::Int=5)
    tmin = Inf
    for _ in 1:reps
        GC.gc()                     # réduit le bruit
        t = @elapsed f()
        if t < tmin
            tmin = t
        end
    end
    return tmin
end

# -------------------------
# Cas synthétique (données + poids)
# -------------------------
function gen_case(; n=64, d=32, h=64, task=:reg)
    # Données & poids init
    X = randn(RNG, n, d)
    if task == :reg
        y = randn(RNG, n)
        act = identity_activation           # ou tanh_activation / relu_activation
        lossfun = mse
    else
        # classification
        k = 3
        y_ids = rand(RNG, 1:k, n)
        y = one_hot_encode(y_ids)
        act = relu_softmax
        lossfun = cross_entropy
    end
    w = random_weights(X, y, h)
    return (X=X, y=y, w=w, act=act, lossfun=lossfun)
end

# -------------------------
# 1) BENCHMARK part1
# -------------------------
function bench_grad_once(; n=64, d=32, h=64, task=:reg, reps=5)
    case = gen_case(n=n, d=d, h=h, task=task)
    L = loss_of(case.lossfun, case.act, case.X, case.y)
    w = case.w
    N = nparams(w)

    # warmup
    Forward.gradient(L, deepcopy(w))
    VectReverse.gradient(L, deepcopy(w))

    t_fwd = time_min(; reps=reps) do
        Forward.gradient(L, deepcopy(w))
    end
    t_rev = time_min(; reps=reps) do
        VectReverse.gradient(L, deepcopy(w))
    end

    bytes_f = @allocated Forward.gradient(L, deepcopy(w))
    bytes_r = @allocated VectReverse.gradient(L, deepcopy(w))

    println("task=$(task) n=$(n) d=$(d) h=$(h) | params=$(N)")
    @printf("  Forward.gradient:  %.3e s   (alloc ~ %.2f MB)\n",
            t_fwd, bytes_f/1e6)
    @printf("  Reverse.gradient:  %.3e s   (alloc ~ %.2f MB)\n",
            t_rev, bytes_r/1e6)
    @printf("  speedup (Forward / Reverse) ≈ %.2fx\n", t_fwd/t_rev)
    println()
end


# vector (plat) -> Flatten avec même structure que `template`

function vector_to_flatten(v::Vector{Float64}, template::Flatten)
    out = similar(template, Float64)  
    off = 1
    for i in eachindex(template.components)
        blk = template.components[i]
        n = length(blk)
        @inbounds out.components[i] .= reshape(v[off:off+n-1], size(blk))
        off += n
    end
    return out
end

# Wrapper pratique
loss_of(lossfun, act, X, y) = loss(lossfun, act, X, y)

# -------------------------
# 2) BENCHMARK HVP part2
# -------------------------
function bench_hvp_once(; n=64, d=32, h=64, task=:reg, reps=3)
    case = gen_case(n=n, d=d, h=h, task=task)
    L = loss_of(case.lossfun, case.act, case.X, case.y)
    N = nparams(case.w)

    v  = randn(RNG, N)
    vF = vector_to_flatten(v, case.w)

    # warmups (hors mesure)
    Forward.hvp(L, deepcopy(case.w), deepcopy(vF))
    VectReverse.hvp(L, deepcopy(case.w), v)

    # timings
    t_fwd = time_min(; reps=reps) do
        Forward.hvp(L, deepcopy(case.w), vF)
    end
    t_rev = time_min(; reps=reps) do
        VectReverse.hvp(L, deepcopy(case.w), v)
    end

    return (; n, d, h, task, N,
            t_forward=t_fwd, t_rof=t_rev,
            speedup_forward_over_rof=(t_fwd/t_rev),
            ratio=t_rev/t_fwd)
end

function bench_hvp_suite()
    configs = [
        (n=32, d=16, h=16),
        (n=64, d=32, h=32),
    ]
    tasks = [:reg, :class]

    results = []
    for c in configs, t in tasks
        push!(results, bench_hvp_once(; n=c.n, d=c.d, h=c.h, task=t))
    end

    println("=== HVP benchmark (régression et classification) ===")
    for r in results
        @printf "task=%-6s n=%-4d d=%-4d h=%-4d | params=%-7d Forward=%.3e s  RoF=%.3e s  speedup=%.2fx\n" string(r.task) r.n r.d r.h r.N r.t_forward r.t_rof r.speedup_forward_over_rof
    end
    println()
    return results
end



function mem_profile_hvp(L, w; taskname="Forward", v_flat::Union{Nothing,Vector{Float64}}=nothing)
    if v_flat === nothing
        v_flat = randn(length(w))
    end
 
    vF = vector_to_flatten(v_flat, w)
    bytes_f = @allocated Forward.hvp(L, deepcopy(w), vF)
    bytes_r = @allocated VectReverse.hvp(L, deepcopy(w), v_flat)
    println("  [alloc] Forward.hvp: $(round(bytes_f/1e6; digits=2)) MB   |   RoF.hvp: $(round(bytes_r/1e6; digits=2)) MB")
end

# --- un petit runner qui affiche proprement un benchmark simple
function run_one_bench(; n=64, d=32, h=64, task=:reg, reps=5, show_alloc=false)
    case = gen_case(n=n, d=d, h=h, task=task)
    L = loss_of(case.lossfun, case.act, case.X, case.y)
    N = nparams(case.w)
    v  = randn(RNG, N)
    vF = vector_to_flatten(v, case.w)

    
    Forward.hvp(L, deepcopy(case.w), deepcopy(vF))
    VectReverse.hvp(L, deepcopy(case.w), v)

    tf = time_min(; reps=reps) do
        Forward.hvp(L, deepcopy(case.w), vF)
    end
    tr = time_min(; reps=reps) do
        VectReverse.hvp(L, deepcopy(case.w), v)
    end

    println("task=$(task)  n=$(n) d=$(d) h=$(h)  | params=$(N)  Forward=$(tf) s  RoF=$(tr) s  speedup=$(tf/tr)x")
    if show_alloc
        mem_profile_hvp(L, case.w; taskname=string(task), v_flat=v)
    end
end


# -------------------------
# 2) TRAINING : GD vs. Newton-CG (avec seulement H*v)
# -------------------------
grad_ref(f, w) = Forward.gradient(f, w)            # gradient référence
hvp_rof(f, w, v) = VectReverse.hvp(f, w, v)        # H*v via RoF (ton code)

# Conjugate Gradient tronqué sur H p = -g 
function tncg(f, w; g = grad_ref(f, w),
              tol=1e-4, cg_maxit=30)
    gvec = flatten_vec(g)
    nrm_g = norm(gvec)
    nrm_g ≤ tol && return zero(w), nrm_g, 0

    # opérateur linéaire v ↦ H*v
    function Hv(v)
        hv = hvp_rof(f, w, v)
        return hv
    end

    b = -gvec
    x = zeros(eltype(b), length(b))
    r = b - Hv(x)
    p = copy(r)
    rsold = dot(r, r)

    it = 0
    for k in 1:cg_maxit
        Hp = Hv(p)
        alpha = rsold / (dot(p, Hp) + 1e-12)
        x .+= alpha .* p
        r .-= alpha .* Hp
        rsnew = dot(r, r)
        it = k
        if sqrt(rsnew) < 1e-6
            break
        end
        p .= r .+ (rsnew/rsold) .* p
        rsold = rsnew
    end

   
    dx = similar(w, Float64)
    offset = 1
    for i in eachindex(w.components)
        n = length(w.components[i])
        dx.components[i] .= reshape(x[offset:offset+n-1], size(w.components[i]))
        offset += n
    end

    return dx, nrm_g, it
end

# Boucle d’entraînement : GD vs Newton-CG
function train_compare(; n=64, d=16, h=16, task=:reg, iters=5, ne=1e-2)
    case = gen_case(n=n, d=d, h=h, task=task)
    L = loss_of(case.lossfun, case.act, case.X, case.y)

    w_gd = deepcopy(case.w)
    w_nc = deepcopy(case.w)

    println("=== Entraînement (task=$(task))  n=$n d=$d h=$h ===")

    # --- GD ---
    @time begin
        for _ in 1:iters
            g = Forward.gradient(L, w_gd)
            for i in eachindex(w_gd.components)
                w_gd.components[i] .-= ne .* g.components[i]
            end
        end
    end
    loss_gd = L(w_gd)
    println("GD:         loss = $(loss_gd) after $iters iters.")

    # --- Newton-CG (H*v uniquement) ---
    @time begin
        for _ in 1:iters
            g = Forward.gradient(L, w_nc)
            step, _, _ = tncg(L, w_nc; g=g, cg_maxit=20)  # CG plus court
            # line search simple
            alpha = 1.0
            old = L(w_nc)
            while true
                trial = deepcopy(w_nc)
                for i in eachindex(trial.components)
                    trial.components[i] .+= alpha .* step.components[i]
                end
                if L(trial) ≤ old || alpha < 1e-6
                    w_nc = trial
                    break
                end
                alpha *= 0.5
            end
        end
    end
    loss_nc = L(w_nc)
    println("Newton-CG:  loss = $(loss_nc) after $iters iters. (H*v only)")

    return (; loss_gd, loss_nc)
end

# ===========================
# RUN
# ===========================
println("=== gradient benchmark (régression) ===")
bench_grad_once(; n=32, d=16, h=16, task=:reg)
bench_grad_once(; n=64, d=32, h=32, task=:reg)
println("=== gradient benchmark (classification) ===")
bench_grad_once(; n=32, d=16, h=16, task=:class)

println("=== HVP benchmark (régression) ===")
run_one_bench(; n=32, d=16, h=16, task=:reg, reps=5, show_alloc=true)
run_one_bench(; n=64, d=32, h=32, task=:reg, reps=5, show_alloc=true)

println("\n=== Entraînement (task=reg) ===")
train_compare(; n=64, d=16, h=16, task=:reg, iters=5, ne=1e-2)

println("\n=== HVP benchmark (classification) ===")
run_one_bench(; n=64, d=32, h=32, task=:class, reps=8, show_alloc=true)

println("\n=== Entraînement (task=class) ===")
train_compare(; n=64, d=16, h=16, task=:class, iters=5, ne=5e-3)

println("done.")
