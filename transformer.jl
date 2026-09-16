module Transformer
using Random

using ..VectReverse: VectNode, softmax, relu, layernorm
import ..Flatten

export init_transformer_params, forward_loss_fn, flatten_params, unflatten_params


function ensure_vectnode(x)
    return x isa VectNode ? x : VectNode(x)
end

# Initialization of parameters (MULTI-LAYER)
function init_transformer_params(vocab::Int, d_model::Int, d_ff::Int, max_seq_len::Int; n_layers=1, rng=Random.GLOBAL_RNG)
    scale_emb = sqrt(1.0 / d_model)
    scale_attn = sqrt(1.0 / d_model)
    scale_ff_in = sqrt(2.0 / d_model)
    scale_ff_out = sqrt(2.0 / d_ff)
    scale_out = sqrt(1.0 / d_model)
    
    params = Dict{Symbol, VectNode}()
    
    # Embeddings 
    params[:Wemb] = VectNode(scale_emb * randn(rng, Float64, vocab, d_model))
    params[:Wpos] = VectNode(scale_emb * randn(rng, Float64, max_seq_len, d_model))
    
    params[:n_layers] = VectNode([Float64(n_layers)])
    
    for layer in 1:n_layers
        params[Symbol("Wq_$layer")] = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
        params[Symbol("Wk_$layer")] = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
        params[Symbol("Wv_$layer")] = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
        params[Symbol("Wo_$layer")] = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
        params[Symbol("W1_$layer")] = VectNode(scale_ff_in * randn(rng, Float64, d_model, d_ff))
        params[Symbol("b1_$layer")] = VectNode(zeros(Float64, 1, d_ff))
        params[Symbol("W2_$layer")] = VectNode(scale_ff_out * randn(rng, Float64, d_ff, d_model))
        params[Symbol("b2_$layer")] = VectNode(zeros(Float64, 1, d_model))
    end
    
    # Output layer
    params[:Wout] = VectNode(scale_out * randn(rng, Float64, d_model, vocab))
    params[:bout] = VectNode(zeros(Float64, 1, vocab))
    
    return params
end

# Flatten/Unflatten 

function flatten_params(p::Dict{Symbol, VectNode})
    comps = Any[]
    
    push!(comps, p[:Wemb].value)
    push!(comps, p[:Wpos].value)
    
    n_layers = Int(p[:n_layers].value[1])
    push!(comps, p[:n_layers].value)
    
    for layer in 1:n_layers
        push!(comps, p[Symbol("Wq_$layer")].value)
        push!(comps, p[Symbol("Wk_$layer")].value)
        push!(comps, p[Symbol("Wv_$layer")].value)
        push!(comps, p[Symbol("Wo_$layer")].value)
        push!(comps, p[Symbol("W1_$layer")].value)
        push!(comps, p[Symbol("b1_$layer")].value)
        push!(comps, p[Symbol("W2_$layer")].value)
        push!(comps, p[Symbol("b2_$layer")].value)
    end
    
    push!(comps, p[:Wout].value)
    push!(comps, p[:bout].value)
    
    return Flatten(comps)
end

function unflatten_params(f::Flatten)
    comps = f.components
    params = Dict{Symbol, VectNode}()
    
    function to_vectnode(x)
        if x isa VectNode
            return x
        else
            return VectNode(x)
        end
    end
    
    params[:Wemb] = to_vectnode(comps[1])
    params[:Wpos] = to_vectnode(comps[2])
    
    n_layers_comp = comps[3]
    n_layers = if n_layers_comp isa VectNode
        Int(n_layers_comp.value[1])
    else
        Int(n_layers_comp[1])
    end
    params[:n_layers] = to_vectnode(n_layers_comp)
    
    idx = 4
    for layer in 1:n_layers
        params[Symbol("Wq_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("Wk_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("Wv_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("Wo_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("W1_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("b1_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("W2_$layer")] = to_vectnode(comps[idx]); idx += 1
        params[Symbol("b2_$layer")] = to_vectnode(comps[idx]); idx += 1
    end
    
    params[:Wout] = to_vectnode(comps[idx])
    params[:bout] = to_vectnode(comps[idx+1])
    
    return params
end

# Embeddings

function one_hot_node(idx::Int, vocab::Int)
    v = zeros(Float64, 1, vocab)
    v[1, idx] = 1.0
    return VectNode(v)
end

function embed_sequence(seq::Vector{Int}, Wemb::VectNode, Wpos::VectNode)
    seq_len = length(seq)
    d_model = size(Wemb.value, 2)
    
    embedded = VectNode(zeros(Float64, seq_len, d_model))
    
    for (i, token_idx) in enumerate(seq)
        token_emb = one_hot_node(token_idx, size(Wemb.value, 1)) * Wemb
        pos_emb = VectNode(Wpos.value[i:i, :])
        combined = token_emb + pos_emb
        
        if i == 1
            embedded = combined
        else
            embedded = VectNode(:vcat, [embedded, combined], 
                               vcat(embedded.value, combined.value))
        end
    end
    
    return embedded
end

# Masque causal
function create_causal_mask(seq_len::Int)
    mask = zeros(Float64, seq_len, seq_len)
    for i in 1:seq_len
        for j in 1:seq_len
            if j > i
                mask[i, j] = -1e9
            end
        end
    end
    return mask
end

# Attention
function transpose_node(x::VectNode)
    return VectNode(:transpose, [x], x.value')
end

function masked_scaled_dot_attention(Q::VectNode, K::VectNode, V::VectNode, mask::Matrix{Float64})
    d = size(Q.value, 2)
    score = (Q * transpose_node(K)) / sqrt(d)
    score_masked = score + VectNode(mask)
    weights = softmax(score_masked; dims=2)
    return weights * V
end

# Feedforward

function feed_forward(x::VectNode, W1::VectNode, b1::VectNode,
                      W2::VectNode, b2::VectNode)
    h = x * W1 .+ b1
    hact = relu(h)
    out = hact * W2 .+ b2
    return out
end

function linear_to_vocab(x::VectNode, Wout::VectNode, bout::VectNode)
    return x * Wout .+ bout
end

# Multi-Head Attention
function split_heads(x::VectNode, n_heads::Int)
    seq_len, d_model = size(x.value)
    d_head = div(d_model, n_heads)
    heads = []
    
    for i in 0:(n_heads-1)
        start_idx = i * d_head + 1
        end_idx = (i + 1) * d_head
        slice = x.value[:, start_idx:end_idx]
        
        head_node = VectNode(:slice, [x], slice)
        head_node.params[:slice_range] = (start_idx, end_idx)
        head_node.params[:n_heads] = n_heads
        head_node.params[:head_index] = i
        
        push!(heads, head_node)
    end
    return heads
end

function concat_heads(heads::Vector{VectNode})
    concat_node = VectNode(:concat, heads, hcat([h.value for h in heads]...))
    return concat_node
end

function multihead_attention(x::VectNode, Wq::VectNode, Wk::VectNode, Wv::VectNode,
                             Wo::VectNode, mask::Matrix{Float64}; n_heads=2)
    Q = x * Wq
    K = x * Wk
    V = x * Wv
    
    Q_heads = split_heads(Q, n_heads)
    K_heads = split_heads(K, n_heads)
    V_heads = split_heads(V, n_heads)
    
    head_outputs = VectNode[]
    for i in 1:n_heads
        h = masked_scaled_dot_attention(Q_heads[i], K_heads[i], V_heads[i], mask)
        push!(head_outputs, h)
    end
    
    concat = concat_heads(head_outputs)
    return concat * Wo
end

# Bloc Transformer
function transformer_block(x::VectNode, Wq::VectNode, Wk::VectNode, Wv::VectNode,
                          Wo::VectNode, W1::VectNode, b1::VectNode,
                          W2::VectNode, b2::VectNode, mask::Matrix{Float64}; n_heads=2)
    attn_out = multihead_attention(x, Wq, Wk, Wv, Wo, mask; n_heads=n_heads)
    x = layernorm(x + attn_out; eps=1e-5)
    
    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layernorm(x + ff_out; eps=1e-5)
    
    return x
end

# Cross-Entropy Loss
function cross_entropy_loss(logits::VectNode, target_idx::Int, vocab::Int)
    probs = softmax(logits; dims=2)
    log_probs = log.(probs .+ VectNode(1e-10))
    target_onehot = one_hot_node(target_idx, vocab)
    loss = -(target_onehot .* log_probs)
    return sum(loss)
end

# Loss function factory 
function forward_loss_fn(batch_inputs::Vector{Vector{Int}}, batch_targets::Vector{Vector{Int}}; n_heads=2)
    function loss_fn(params_nodes::Flatten)
        p = unflatten_params(params_nodes)
        
        n_layers = if p[:n_layers] isa VectNode
            Int(p[:n_layers].value[1])
        else
            Int(p[:n_layers][1])
        end
        
        Wemb = p[:Wemb]
        Wpos = p[:Wpos]
        Wout = p[:Wout]
        bout = p[:bout]
        
        total_loss = VectNode(0.0)
        B = length(batch_inputs)
        
        for i in 1:B
            seq = batch_inputs[i]
            tgt_seq = batch_targets[i]
            seq_len = length(seq)
            vocab = size(Wout.value, 2)
            
            x = embed_sequence(seq, Wemb, Wpos)
            mask = create_causal_mask(seq_len)
            
            for layer in 1:n_layers
                x = transformer_block(
                    x,
                    p[Symbol("Wq_$layer")], p[Symbol("Wk_$layer")],
                    p[Symbol("Wv_$layer")], p[Symbol("Wo_$layer")],
                    p[Symbol("W1_$layer")], p[Symbol("b1_$layer")],
                    p[Symbol("W2_$layer")], p[Symbol("b2_$layer")],
                    mask; n_heads=n_heads
                )
            end
            
            logits = linear_to_vocab(x, Wout, bout)
            
            sample_loss = VectNode(0.0)
            for t in 1:seq_len
                if t <= length(tgt_seq)
                    logits_t = VectNode(:slice, [logits], logits.value[t:t, :])
                    logits_t.params[:slice_range] = (t, t)
                    ce = cross_entropy_loss(logits_t, tgt_seq[t], vocab)
                    sample_loss = sample_loss + ce
                end
            end
            
            sample_loss = sample_loss / seq_len
            total_loss = total_loss + sample_loss
        end
        
        return total_loss / B
    end
    return loss_fn
end

end 