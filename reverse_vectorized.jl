module VectReverse
export VectNode, softmax, relu
mutable struct VectNode
    op::Union{Symbol,Nothing}
    args::Vector{VectNode}
    value::Union{Number, AbstractArray}
    derivative::Union{Number, AbstractArray}
    tangent::Union{Number, AbstractArray}
    params::Dict{Symbol, Any}
end

import ..Flatten

function VectNode(value::Union{Number, AbstractArray})
    VectNode(nothing, VectNode[], value, zero(value), zero(value), Dict{Symbol, Any}())
end

function VectNode(op, args, value)
    derivative = isa(value, Number) ? 0.0 : zeros(Float64, size(value))
    tangent = isa(value, Number) ? 0.0 : zeros(Float64, size(value))
    return VectNode(op, args, value, derivative, tangent, Dict{Symbol, Any}())
end

Base.size(x::VectNode) = size(x.value)
Base.length(x::VectNode) = length(x.value) 
Base.ndims(x::VectNode) = ndims(x.value)

# ═══════════════════════════════════════════════════════════
# BROADCASTING
# ═══════════════════════════════════════════════════════════

# Unaire: f.(x)
function Base.broadcasted(op::Function, x::VectNode)
    result_val = op.(x.value)
    op_sym = Symbol(op)
    return VectNode(op_sym, [x], result_val)
end

# Binaire: f.(x, y) avec (VectNode, VectNode)
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    result = broadcast(op, x.value, y.value)
    op_sym = op === (*) ? :hadamard : Symbol(op)
    return VectNode(op_sym, [x, y], result)
end

# Binaire: f.(x, y) avec (VectNode, Array/Number)
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    result = broadcast(op, x.value, y)
    op_sym = op === (*) ? :hadamard : Symbol(op)
    return VectNode(op_sym, [x, VectNode(y)], result)
end

# Binaire: f.(x, y) avec (Array/Number, VectNode)
function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    result = broadcast(op, x, y.value)
    op_sym = op === (*) ? :hadamard : Symbol(op)
    return VectNode(op_sym, [VectNode(x), y], result)
end

# Puissance avec littéral: x .^ Val(y)
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
    Base.broadcasted(^, x, y)
end

# ═══════════════════════════════════════════════════════════
# OPÉRATIONS ARITHMÉTIQUES
# ═══════════════════════════════════════════════════════════
Base.:+(x::VectNode, y::VectNode) = VectNode(:+, [x, y], x.value + y.value)
Base.:+(x::VectNode, y::Number)   = VectNode(:+, [x, VectNode(y)], x.value + y)
Base.:+(x::Number,   y::VectNode) = VectNode(:+, [VectNode(x), y], x + y.value)

Base.:-(x::VectNode, y::VectNode) = VectNode(:-, [x, y], x.value - y.value)
Base.:-(x::VectNode, y::Number)   = VectNode(:-, [x, VectNode(y)], x.value - y)
Base.:-(x::Number,   y::VectNode) = VectNode(:-, [VectNode(x), y], x - y.value)
Base.:-(x::VectNode, y::AbstractArray) = VectNode(:-, [x, VectNode(y)], x.value .- y)
Base.:-(x::AbstractArray, y::VectNode) = VectNode(:-, [VectNode(x), y], x .- y.value)
Base.:-(x::VectNode) = VectNode(:-, [x], -x.value)


Base.:*(x::VectNode, y::VectNode) = begin
    xv, yv = x.value, y.value
    result = if isa(xv, AbstractVector) && isa(yv, AbstractVector)
        xv' * yv                    # dot → scalaire
    else
        xv * yv                     # matmul ou scalaire*array
    end
    VectNode(:*, [x, y], result)
end
Base.:*(x::VectNode, y::Number)        = VectNode(:*, [x, VectNode(y)], x.value * y)
Base.:*(x::Number,   y::VectNode)      = VectNode(:*, [VectNode(x), y], x * y.value)
Base.:*(x::AbstractArray, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)
Base.:*(x::VectNode, y::AbstractArray) = VectNode(:*, [x, VectNode(y)], x.value * y)

Base.:/(x::VectNode, y::VectNode) = VectNode(:/, [x, y], x.value / y.value)
Base.:/(x::VectNode, y::Number)   = VectNode(:/, [x, VectNode(y)], x.value / y)
Base.:/(x::Number,   y::VectNode) = VectNode(:/, [VectNode(x), y], x / y.value)



function Base.sum(x::VectNode; dims=nothing)
    if isnothing(dims)
        return VectNode(:sum, [x], sum(x.value))
    else
        node = VectNode(:sum_dims, [x], sum(x.value, dims=dims))
        node.params[:dims] = dims
        return node
    end
end


function Base.maximum(x::VectNode; dims=nothing)
    if isnothing(dims)
        return VectNode(:maximum, [x], maximum(x.value))
    else
        node = VectNode(:maximum, [x], maximum(x.value, dims=dims))
        node.params[:dims] = dims              
        return node
    end
end


function relu(x::VectNode)
    return VectNode(:relu, [x], max.(0, x.value))
end

function softmax(x::VectNode; dims=1)
    # x.value est toujours Number ou Array
    exps_val = exp.(x.value .- maximum(x.value; dims=dims))
    s_val = sum(exps_val; dims=dims)
    result_val = exps_val ./ s_val
    
    return VectNode(:softmax, [x], result_val)  # value = Array ou Number
end


function layernorm(x::VectNode; eps=1e-5)
    μ = sum(x.value)/length(x.value)
    sigma2 = sum((x.value .- μ).^2)/length(x.value)
    normalized = (x.value .- μ) ./ sqrt(sigma2 + eps)
    return VectNode(:layernorm, [x], normalized)
end

function Base.transpose(x::VectNode)
    return VectNode(:transpose, [x], x.value')
end

Base.exp(x::VectNode) = VectNode(:exp, [x], exp.(x.value))

function Base.broadcasted(::typeof(log), x::VectNode)
    result_val = log.(x.value)
    return VectNode(:log, [x], result_val)
end

function Base.zero(x::Flatten)
    zero_components = []
    for comp in x.components
        if isa(comp, VectNode)
            push!(zero_components, zero(comp.value))
        else
            push!(zero_components, zero(comp))
        end
    end
    return Flatten(zero_components)
end


function topo_sort(f::VectNode)
    visited = Set{VectNode}()
    order = VectNode[]
    function _dfs(node::VectNode)
        if node ∉ visited
            push!(visited, node)
            for arg in node.args
                _dfs(arg)
            end
            push!(order, node)
        end
    end
    _dfs(f)
    return order
end

# ═══════════════════════════════════════════════════════════
# BACKWARD PASS
# ═══════════════════════════════════════════════════════════

function backward!(f::VectNode)
    topo = topo_sort(f)

    # 1) Init
    for node in topo
        if isa(node.value, Number)
            node.derivative = 0.0
        else
            node.derivative = zeros(Float64, size(node.value))
        end
    end

    # 2) Gradient de sortie
    if isa(f.value, Number)
        f.derivative = 1.0
    elseif length(f.value) == 1
        f.derivative = 1.0
    else
        f.derivative = ones(Float64, size(f.value))
    end

    # 3) Backprop
    for node in reverse(topo)
        _backward!(node)
    end

    return f
end


function _backward!(node::VectNode)
    if isnothing(node.op) || node.op == :scalar
        return
    end

    # Addition
    if node.op == :+
        for arg in node.args
            if isa(arg.derivative, Number) && isa(node.derivative, Number)
                # Scalaire + Scalaire
                arg.derivative += node.derivative
            elseif isa(arg.derivative, Number) && isa(node.derivative, AbstractArray)
                # arg est scalaire, node.derivative est array
                arg.derivative += sum(node.derivative)
            elseif isa(arg.derivative, AbstractArray) && isa(node.derivative, Number)
                # arg est array, node.derivative est scalaire
                arg.derivative .+= node.derivative
            else
                # Array + Array (peut nécessiter du broadcasting)
                if size(arg.derivative) == size(node.derivative)
                    # Mêmes dimensions : addition directe
                    arg.derivative .+= node.derivative
                else
                    # Dimensions différentes : réduire les dimensions broadcastées
                    grad_to_add = node.derivative
                    for dim in 1:ndims(grad_to_add)
                        if size(arg.derivative, dim) == 1 && size(grad_to_add, dim) > 1
                            grad_to_add = sum(grad_to_add, dims=dim)
                        end
                    end
                    arg.derivative .+= grad_to_add
                end
            end
        end

    # Soustraction
    elseif node.op == :-
        if length(node.args) == 1
            
            x = node.args[1]
            if isa(x.derivative, Number)
                x.derivative += -node.derivative
            else
                x.derivative .+= .-node.derivative
            end
        else
            
            x, y = node.args
            
            
            if size(x.value) == size(node.derivative)
                x.derivative .+= node.derivative
            else
                axes_to_reduce = tuple([i for i in 1:ndims(node.derivative) if size(x.value, i) == 1]...)
                if isempty(axes_to_reduce)
                    x.derivative .+= node.derivative
                else
                    x.derivative .+= sum(node.derivative, dims=axes_to_reduce)
                end
            end
            
            
            grad_y = .-node.derivative
            if size(y.value) == size(grad_y)
                y.derivative .+= grad_y
            else
                axes_to_reduce = tuple([i for i in 1:ndims(grad_y) if size(y.value, i) == 1]...)
                if isempty(axes_to_reduce)
                    y.derivative .+= grad_y
                else
                    y.derivative .+= sum(grad_y, dims=axes_to_reduce)
                end
            end
        end

    elseif node.op == :hadamard
        x, y = node.args
        G = node.derivative
        
        grad_x = G .* y.value
        grad_y = G .* x.value
        
    
        x.derivative .+= grad_x
        y.derivative .+= grad_y

    elseif node.op == :dot
        x, y = node.args
        G = node.derivative
        x.derivative .+= G * y.value
        y.derivative .+= G * x.value

    # Multiplication
    elseif node.op == :*
        x, y = node.args
        xv, yv = x.value, y.value
        grad = node.derivative

        if isa(xv, Number) && isa(yv, Number)
            x.derivative += grad * yv
            y.derivative += grad * xv
        elseif isa(xv, AbstractVector) && isa(yv, AbstractVector)
            x.derivative .+= grad * yv
            y.derivative .+= grad * xv
        elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
            x.derivative .+= grad * yv'
            y.derivative .+= xv' * grad
        elseif isa(xv, AbstractArray) && isa(yv, Number)
            x.derivative .+= grad .* yv
            y.derivative += sum(grad .* xv)
        elseif isa(xv, Number) && isa(yv, AbstractArray)
            x.derivative += sum(grad .* yv)
            y.derivative .+= grad .* xv
        end

    # Division
    elseif node.op == :/
        x, y = node.args
        
        if isa(x.derivative, Number)
            x.derivative += node.derivative / (y.value)
        else
            x.derivative .+= node.derivative ./ (y.value)
        end
        
        expr = node.derivative .* (.-x.value ./ (y.value.^2))
        if isa(y.derivative, Number)
            y.derivative += sum(expr)
        else
            if size(y.derivative) != size(expr)
                dims_to_sum = findall(size(expr) .> size(y.derivative))
                y.derivative .+= sum(expr, dims=tuple(dims_to_sum...))
            else
                y.derivative .+= expr
            end
        end

    # Tanh
    elseif node.op == :tanh
        x = node.args[1]
        if isa(x.derivative, Number)
            x.derivative += node.derivative * (1 - node.value^2)
        else
            x.derivative .+= node.derivative .* (1 .- node.value.^2)
        end

    # Puissance
    elseif node.op == :^
        x = node.args[1]
        n = node.args[2]
        if isa(x.derivative, Number)
            x.derivative += node.derivative * n.value * (x.value^(n.value - 1))
        else
            x.derivative .+= node.derivative .* n.value .* (x.value.^(n.value .- 1))
        end
    
    # Sum
    elseif node.op == :sum
        x = node.args[1]
        if isa(x.derivative, Number)
            x.derivative += node.derivative
        else
            x.derivative .+= node.derivative
        end

    # Sum avec dimensions
    elseif node.op == :sum_dims
        x = node.args[1]
        
        if isa(x.derivative, Number)
            x.derivative += sum(node.derivative)
        else
            
            x.derivative .+= node.derivative
        end

    # Maximum
    elseif node.op == :maximum
        x = node.args[1]
        if isa(x.derivative, Number)
            mask = (x.value == node.value)
            x.derivative += node.derivative * (mask ? 1.0 : 0.0)
        else
            mask = float.(x.value .== node.value)
            x.derivative .+= node.derivative .* mask
        end

    # ReLU
    elseif node.op == :relu
        x = node.args[1]
        if isa(x.derivative, Number)
            mask = (x.value > 0)
            x.derivative += node.derivative * (mask ? 1.0 : 0.0)
        else
            mask = float.(x.value .> 0)
            x.derivative .+= node.derivative .* mask
        end
    elseif node.op == :softmax
        x = node.args[1]
        y = node.value
        # Jacobien-vector product pour softmax
        if isa(y, Number)
            x.derivative += 0.0 # trivial case
        else
            # y est un vecteur
            s = y
            g = node.derivative
            # Grad softmax: J*v = s .* (g - sum(s .* g))
            x.derivative .+= s .* (g .- sum(s .* g))
        end
    
        # Transpose
    elseif node.op == :transpose
        x = node.args[1]
        # node.derivative a la forme transposée ; on remet la dérivée sur x en transposant
        if isa(x.derivative, Number)
            # si x est scalaire (rare), on somme tous les éléments de node.derivative
            if isa(node.derivative, Number)
                x.derivative += node.derivative
            else
                x.derivative += sum(node.derivative)
            end
        else
            # propagation simple : d(x) += (d(node))'
            x.derivative .+= node.derivative'
        end

    # Slice (pour split_heads)
    elseif node.op == :slice
        x = node.args[1]
        if haskey(node.params, :slice_range)
            start_idx, end_idx = node.params[:slice_range]
            
            # Déterminer si c'est un slice de lignes ou de colonnes
            if ndims(x.value) == 2 && ndims(node.value) == 2
                # Cas 1: slice de colonnes (split_heads) - x[:, start:end]
                if size(x.value, 1) == size(node.value, 1) && size(x.value, 2) > size(node.value, 2)
                    # S'assurer que les dimensions correspondent
                    if size(node.derivative) == (size(x.value, 1), end_idx - start_idx + 1)
                        x.derivative[:, start_idx:end_idx] .+= node.derivative
                    else
                        # Fallback sécurisé
                        x.derivative[:, start_idx:end_idx] .+= reshape(node.derivative, 
                                                                        size(x.derivative[:, start_idx:end_idx]))
                    end
                # Cas 2: slice de lignes (extraction de position) - x[start:end, :]
                elseif size(x.value, 2) == size(node.value, 2) && size(x.value, 1) > size(node.value, 1)
                    if size(node.derivative) == (end_idx - start_idx + 1, size(x.value, 2))
                        x.derivative[start_idx:end_idx, :] .+= node.derivative
                    else
                        # Fallback sécurisé
                        x.derivative[start_idx:end_idx, :] .+= reshape(node.derivative,
                                                                        size(x.derivative[start_idx:end_idx, :]))
                    end
                else
                    # Cas générique: accumuler en sommant si les dimensions ne matchent pas
                    if isa(x.derivative, Number)
                        x.derivative += sum(node.derivative)
                    else
                        # Essayer d'accumuler directement
                        try
                            x.derivative[:, start_idx:end_idx] .+= node.derivative
                        catch
                            try
                                x.derivative[start_idx:end_idx, :] .+= node.derivative
                            catch
                                # Dernier recours: sommer tout
                                x.derivative .+= sum(node.derivative)
                            end
                        end
                    end
                end
            elseif ndims(x.value) == 1
                # Vecteur 1D
                x.derivative[start_idx:end_idx] .+= node.derivative
            else
                # Fallback pour autres cas
                if isa(x.derivative, Number)
                    x.derivative += sum(node.derivative)
                else
                    x.derivative .+= sum(node.derivative)
                end
            end
        else
            # Pas d'indices stockés: accumulation simple
            if isa(x.derivative, Number)
                x.derivative += sum(node.derivative)
            else
                # Essayer d'accumuler en broadcastant
                if size(x.derivative) == size(node.derivative)
                    x.derivative .+= node.derivative
                else
                    x.derivative .+= sum(node.derivative)
                end
            end
        end

    # Concat
    elseif node.op == :concat
        # Redistribuer le gradient aux têtes
        start_col = 1
        for head in node.args
            d_head = size(head.value, 2)
            grad_slice = node.derivative[:, start_col:(start_col + d_head - 1)]
            if isa(head.derivative, Number)
                head.derivative += sum(grad_slice)
            else
                head.derivative .+= grad_slice
            end
            start_col += d_head
        end
    elseif node.op == :vcat
        # node.args contient les morceaux à concaténer verticalement
        # node.derivative a la forme (total_rows x d_model)
        start_row = 1
        for piece in node.args
            n_rows = size(piece.value, 1)
            # Extraire la tranche du gradient
            grad_slice = node.derivative[start_row:(start_row + n_rows - 1), :]
            
            # Accumuler
            if isa(piece.derivative, Number)
                piece.derivative += sum(grad_slice)
            else
                piece.derivative .+= grad_slice
            end
            start_row += n_rows
        end
    elseif node.op == :layernorm
        x = node.args[1]
        eps = get(node.params, :eps, 1e-5)
        
        # Recalculer forward pass
        N = length(x.value)
        μ = sum(x.value) / N
        centered = x.value .- μ
        var = sum(centered .^ 2) / N
        std = sqrt(var + eps)
        normalized = centered ./ std
        
        # Gradient (formule Batch Normalization adaptée)
        g_out = node.derivative
        
        # Trois termes de la chaîne de dérivation
        sum_dy = sum(g_out)
        sum_dy_norm = sum(g_out .* normalized)
        
        g_x = (1.0 / (N * std)) .* (
            N .* g_out .- sum_dy .- normalized .* sum_dy_norm
        )
        
        if isa(x.derivative, Number)
            x.derivative += sum(g_x)
        else
            x.derivative .+= g_x
        end
    # Log
    elseif node.op == :log
        x = node.args[1]
        if isa(x.derivative, Number)
            x.derivative += node.derivative / (x.value)
        else
            x.derivative .+= node.derivative ./ (x.value)
        end

    # Exp
    elseif node.op == :exp
        x = node.args[1]
        if isa(x.derivative, Number)
            x.derivative += node.derivative * node.value
        else
            x.derivative .+= node.derivative .* node.value
        end
    
    else
        error("Operation `$(node.op)` not supported yet")
    end
end

import ..Flatten

function gradient!(f, g::Flatten, x::Flatten)
    # Transforme chaque composant en VectNode, mais évite VectNode(VectNode)
    x_nodes = [isa(xi, VectNode) ? xi : VectNode(xi) for xi in x.components]

    # On recrée un Flatten de VectNodes
    x_flat = Flatten(x_nodes)

    # On appelle la loss correctement
    expr = f(x_flat)

    # Backprop
    backward!(expr)

    # Remplir le gradient dans le Flatten g
    for i in eachindex(g.components)
        g.components[i] .= x_nodes[i].derivative
    end

    return g
end


gradient(f, x) = gradient!(f, zero(x), x)

function _reduce_for_broadcast(up, target_shape::Tuple)
    if isa(up, Number)
        return up
    end
    if size(up) == target_shape
        return up
    else
        axes_to_sum = Tuple(i for i in 1:ndims(up) if i > length(target_shape) || target_shape[i] == 1)
        return sum(up, dims=axes_to_sum)
    end
end

function rof_pass(f::VectNode, x_nodes::Vector{VectNode}, v::Vector)
    """
    RoF : calcule H*v par
      1) forward des tangentes de valeurs
      2) reverse sur la linéarisation (adjoints des tangentes)
    Retourne [xi.tangent] pour chaque entrée.
    """

    # ——— Préparation : topo + cache des tangentes de valeurs ———
    order = topo_sort(f)
    dval  = Dict{VectNode,Any}()   

    #  Seeds dval 
    function __prime_dval!()
        # feuilles (constantes) → tangente 0
        for n in order
            if isnothing(n.op)
                dval[n] = zero(n.value)
            end
        end
        # entrées : découper v → ẋ
        idx = 0
        for x in x_nodes
            nx     = length(x.value)
            slice  = v[idx+1:idx+nx]
            seed   = (nx == 1) ? fill(slice[1], size(x.value)) : reshape(slice, size(x.value))
            x.tangent = seed
            dval[x]   = seed
            idx += nx
        end
        return nothing
    end

    #  Forward : propage dval le long du graphe
    function __propagate_dval!()
        for n in order
            if isnothing(n.op); continue; end

            if n.op == :+
                acc = zero(n.value)
                for a in n.args
                    acc = (isa(acc, Number) && isa(dval[a], Number)) ? acc + dval[a] : acc .+ dval[a]
                end
                dval[n] = acc

            elseif n.op == :-
                if length(n.args) == 1
                    x = n.args[1]
                    dval[n] = -dval[x]
                else
                    x, y = n.args
                    dval[n] = dval[x] .- dval[y]
                end

            elseif n.op == :*
                x, y = n.args
                xv, yv = x.value, y.value
                xt, yt = dval[x], dval[y]
                if isa(xv, Number) && isa(yv, Number)
                    dval[n] = xt * yv + xv * yt
                elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                    dval[n] = xt * yv + xv * yt
                else
                    dval[n] = xt .* yv .+ xv .* yt
                end

            elseif n.op == :/
                x, y = n.args
                xv, yv = x.value, y.value
                xt, yt = dval[x], dval[y]
                if isa(xv, Number) && isa(yv, Number)
                    dval[n] = (xt * yv - xv * yt) / (yv^2)
                else
                    dval[n] = (xt .* yv .- xv .* yt) ./ (yv .^ 2)
                end

            elseif n.op == :^
                
                x = n.args[1]; p = 2
                xv, xt = x.value, dval[x]
                dval[n] = isa(xv, AbstractArray) ? (p .* xv .^ (p-1) .* xt) : (p * xv^(p-1) * xt)

            elseif n.op == :tanh
                x = n.args[1]
                dval[n] = (1 .- n.value.^2) .* dval[x]

            elseif n.op == :exp
                x = n.args[1]
                dval[n] = n.value .* dval[x]

            elseif n.op == :log
                x = n.args[1]
                dval[n] = dval[x] ./ x.value

            elseif n.op == :relu
                x = n.args[1]
                dval[n] = float.(x.value .>= 0) .* dval[x]

            elseif n.op == :sum
                x = n.args[1]
                dval[n] = sum(dval[x])

            elseif n.op == :maximum
                x    = n.args[1]
                mask = float.(x.value .== n.value)
                dims = get(n.params, :dims, nothing)
                dval[n] = isnothing(dims) ? sum(dval[x] .* mask) : sum(dval[x] .* mask, dims=dims)

            elseif n.op == :sum_dims
                x    = n.args[1]
                dims = get(n.params, :dims, nothing)
                dval[n] = isnothing(dims) ? sum(dval[x]) : sum(dval[x], dims=dims)

            elseif n.op == :hadamard
                x, y  = n.args
                xv,yv = x.value, y.value
                xt,yt = dval[x], dval[y]
                dval[n] = xt .* yv .+ xv .* yt
            elseif n.op == :transpose
                x = n.args[1]
                # dval[z] = (dval[x])'  because z = x'
                dval[n] = dval[x]'

            else
                error("Opération $(n.op) non implémentée (forward).")
            end
        end
        return nothing
    end

    #  Reset des adjoints RoF (tangentes des adjoints) 
    function __zero_adot!()
        for n in order
            n.tangent = isa(n.value, Number) ? 0.0 : zeros(Float64, size(n.value))
        end
        f.tangent = 0.0
        return nothing
    end

    #  Reverse sur la linéarisation : propage \bar{ż} 
    function __reverse_jvp!()
        for n in reverse(order)
            if isnothing(n.op); continue; end

            bar   = n.derivative  # \bar z (adjoint du primal déjà calculé)
            bar˙  = n.tangent     # \bar{ż} (adjoint de la linéarisation)

            if n.op == :+
                if length(n.args) == 1
                    x = n.args[1]
                    if isa(x.tangent, Number)  x.tangent += bar˙ else x.tangent .+= bar˙ end
                else
                    x, y = n.args
                    if isa(x.tangent, Number)  x.tangent += sum(bar˙) else x.tangent .+= _reduce_for_broadcast(bar˙, size(x.tangent)) end
                    if isa(y.tangent, Number)  y.tangent += sum(bar˙) else y.tangent .+= _reduce_for_broadcast(bar˙, size(y.tangent)) end
                end

            elseif n.op == :-
                if length(n.args) == 1
                    x = n.args[1]
                    if isa(x.tangent, Number)  x.tangent -= bar˙ else x.tangent .-= bar˙ end
                else
                    x, y = n.args
                    if isa(x.tangent, Number)  x.tangent += sum(bar˙) else x.tangent .+= _reduce_for_broadcast(bar˙, size(x.tangent)) end
                    if isa(y.tangent, Number)  y.tangent -= sum(bar˙) else y.tangent .-= _reduce_for_broadcast(bar˙, size(y.tangent)) end
                end

            elseif n.op == :*
                x, y  = n.args
                xv,yv = x.value, y.value
                xt,yt = dval[x], dval[y]
                if isa(xv, Number) && isa(yv, Number)
                    x.tangent += bar˙ * yv + bar * yt
                    y.tangent += xt * bar  + xv * bar˙
                elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                    x.tangent .+= bar˙ * yv' .+ bar * yt'
                    y.tangent .+= xt' * bar  .+ xv' * bar˙
                
                end

            elseif n.op == :/
                x, y  = n.args
                xv,yv = x.value, y.value
                xt,yt = dval[x], dval[y]

                if isa(xv, Number) && isa(yv, Number)
                    x.tangent +=  bar˙ / yv - bar * yt / (yv^2)
                    y.tangent +=  bar˙ * (-xv / yv^2) + bar * ( (2 * xv * yt) / yv^3 - (xt / yv^2) )
                else
                    y2 = yv .^ 2
                    y3 = yv .^ 3
                    # vers x
                    dx = bar˙ ./ yv .- bar .* (yt ./ y2)
                    if !isa(x.tangent, Number)
                        dx = _reduce_for_broadcast(dx, size(x.tangent))
                    end
                    x.tangent .+= dx
                    # vers y
                    dy = (-xv .* bar˙) ./ y2 .+ bar .* ( (2 .* xv .* yt) ./ y3 .- (xt ./ y2) )
                    if !isa(y.tangent, Number)
                        dy = _reduce_for_broadcast(dy, size(y.tangent))
                    end
                    y.tangent .+= dy
                end

            elseif n.op == :^
                x = n.args[1]; p = 2
                xv = x.value
                xt = dval[x]
                if isa(xv, AbstractArray)
                    x.tangent .+= bar˙ .* (p .* xv .^ (p-1)) .+ bar .* (p .* (p-1) .* xv .^ (p-2) .* xt)
                else
                    x.tangent +=  bar˙ * (p * xv^(p-1)) + bar * (p * (p-1) * xv^(p-2) * xt)
                end

            elseif n.op == :tanh
                x  = n.args[1]
                yv = n.value
                xt = dval[x]
                sech2 = 1 .- yv.^2
                x.tangent .+= bar˙ .* sech2 .+ bar .* (-2 .* yv .* sech2 .* xt)

            elseif n.op == :exp
                x  = n.args[1]
                ev = n.value
                xt = dval[x]
                x.tangent .+= bar˙ .* ev .+ bar .* (ev .* xt)

            elseif n.op == :log
                x  = n.args[1]
                xv = x.value
                xt = dval[x]
                x.tangent .+= bar˙ ./ xv .- bar .* (xt ./ (xv .^ 2))

            elseif n.op == :relu
                x    = n.args[1]
                mask = float.(x.value .>= 0)
                x.tangent .+= bar˙ .* mask

            elseif n.op == :sum
                x = n.args[1]
                if isa(bar˙, Number)
                    x.tangent .+= bar˙
                else
                    x.tangent .+= bar˙ .* ones(Float64, size(x.value))
                end

            elseif n.op == :maximum
                x    = n.args[1]
                mask = float.(x.value .== n.value)
                x.tangent .+= bar˙ .* mask

            elseif n.op == :sum_dims
                x = n.args[1]
                if isa(bar˙, Number)
                    x.tangent .+= bar˙
                else
                    x.tangent .+= bar˙ .* ones(Float64, size(x.value))
                end

            elseif n.op == :hadamard
                x, y  = n.args
                xv,yv = x.value, y.value
                xt,yt = dval[x], dval[y]
                tx = bar˙ .* yv .+ bar .* yt
                ty = xt .* bar  .+ xv .* bar˙
                if !isa(x.tangent, Number)
                    tx = (size(tx) == size(x.tangent)) ? tx : _reduce_for_broadcast(tx, size(x.tangent))
                end
                if !isa(y.tangent, Number)
                    ty = (size(ty) == size(y.tangent)) ? ty : _reduce_for_broadcast(ty, size(y.tangent))
                end
                y.tangent .+= ty
                x.tangent .+= tx
            elseif n.op == :transpose
                x = n.args[1]
                # bar˙ est la tangente de z ; pour x on ajoute la transpose
                if isa(x.tangent, Number)
                    # si x scalaire (cas contourné)
                    if isa(bar˙, Number)
                        x.tangent += bar˙
                    else
                        x.tangent += sum(bar˙)
                    end
                else
                    x.tangent .+= bar˙'
                end

            else
                error("Opération $(n.op) non implémentée (reverse).")
            end
        end
        return nothing
    end

    # ——— Exécution ———
    __prime_dval!()
    __propagate_dval!()
    __zero_adot!()
    __reverse_jvp!()

    # Résultat : H*v dans les tangentes des entrées
    return [x.tangent for x in x_nodes]
end


function flatten_tangent(tangents::Vector)
    v = Float64[]
    for t in tangents
        if isa(t, Number)
            push!(v, Float64(t))
        else
            append!(v, vec(t))
        end
    end
    return v
end



function convert_to_flatten(v::AbstractVector, template::Flatten)
    components = []
    offset = 1
    
    for comp in template.components
        if isa(comp, VectNode)
            comp_value = comp.value
        else
            comp_value = comp
        end
        
        n = length(comp_value)
        component_data = v[offset:offset+n-1]
        push!(components, reshape(component_data, size(comp_value)))
        offset += n
    end
    
    return Flatten(components)
end

function hvp(f, x::Flatten, v::Vector)
    """
    Calcule le produit Hessien-vecteur H(x) * v en utilisant l'approche forward-over-reverse.
    v est un vecteur plat de longueur sum(length.(x.components))
    """
    
    x_nodes = [isa(xi, VectNode) ? xi : VectNode(xi) for xi in x.components]
    x_flat = Flatten(x_nodes)
    expr = f(x_flat)
    all_nodes = topo_sort(expr)
    
    #  Passe reverse pour les gradients
    for node in all_nodes
        node.derivative = zero(node.value)
    end
    backward!(expr)
    
    #  Aplatir v selon les composants de x
    v_flat = Float64[]
    offset = 1
    for comp in x.components
        n = length(comp)
        push!(v_flat, v[offset:offset+n-1]...)
        offset += n
    end

    #  Calculer les tangentes
    Hv_result = rof_pass(expr, x_nodes, v_flat)
    
    #  Retourner H*v comme vecteur plat
    return flatten_tangent(Hv_result)
end
function vector_unit(i::Int, template::Flatten)
    n_total = 0
    for comp in template.components
        if isa(comp, VectNode)
            n_total += length(comp.value)
        else
            n_total += length(comp)
        end
    end
    
    @assert 1 ≤ i ≤ n_total "Index $i out of bounds [1, $n_total]"
    
    unit_vec = zeros(Float64, n_total)
    unit_vec[i] = 1.0
    
    return unit_vec  
end

function hessian(f, x::Flatten)
    n = sum(length.(x.components))
    H = zeros(Float64, n, n)
    
    for i in 1:n
        e = vector_unit(i, x) 
        hvp_result = hvp(f, x, e)  
        H[:, i] = flatten_tangent(hvp_result)  # Convertit en Vector
    end
    
    return (H + H') / 2  
end

end