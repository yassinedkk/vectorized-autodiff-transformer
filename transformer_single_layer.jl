module Transformer
using Random

using ..VectReverse: VectNode, softmax, relu, layernorm
import ..Flatten

export init_transformer_params, forward_loss_fn, flatten_params, unflatten_params

function flatten_params(p::Dict{Symbol, VectNode})
    comps = Any[]
    push!(comps, p[:Wemb].value)
    push!(comps, p[:Wpos].value)  
    push!(comps, p[:Wq].value)
    push!(comps, p[:Wk].value)
    push!(comps, p[:Wv].value)
    push!(comps, p[:Wo].value)
    push!(comps, p[:W1].value)
    push!(comps, p[:b1].value)
    push!(comps, p[:W2].value)
    push!(comps, p[:b2].value)
    push!(comps, p[:Wout].value)
    push!(comps, p[:bout].value)
    return Flatten(comps)
end

function unflatten_params(f::Flatten)
    comps = f.components
    return Dict(
        :Wemb => VectNode(comps[1]),
        :Wpos => VectNode(comps[2]),
        :Wq   => VectNode(comps[3]),
        :Wk   => VectNode(comps[4]),
        :Wv   => VectNode(comps[5]),
        :Wo   => VectNode(comps[6]),
        :W1   => VectNode(comps[7]),
        :b1   => VectNode(comps[8]),
        :W2   => VectNode(comps[9]),
        :b2   => VectNode(comps[10]),
        :Wout => VectNode(comps[11]),
        :bout => VectNode(comps[12])
    )
end

function init_transformer_params(vocab::Int, d_model::Int, d_ff::Int, max_seq_len::Int; rng=Random.GLOBAL_RNG)

    scale_emb = sqrt(1.0 / d_model)
    scale_attn = sqrt(1.0 / d_model)
    scale_ff_in = sqrt(2.0 / d_model)   
    scale_ff_out = sqrt(2.0 / d_ff)
    scale_out = sqrt(1.0 / d_model)
    
    Wemb = VectNode(scale_emb * randn(rng, Float64, vocab, d_model))
    Wpos = VectNode(scale_emb * randn(rng, Float64, max_seq_len, d_model))
    Wq   = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
    Wk   = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
    Wv   = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
    Wo   = VectNode(scale_attn * randn(rng, Float64, d_model, d_model))
    W1   = VectNode(scale_ff_in * randn(rng, Float64, d_model, d_ff))
    b1   = VectNode(zeros(Float64, 1, d_ff))
    W2   = VectNode(scale_ff_out * randn(rng, Float64, d_ff, d_model))
    b2   = VectNode(zeros(Float64, 1, d_model))
    Wout = VectNode(scale_out * randn(rng, Float64, d_model, vocab))
    bout = VectNode(zeros(Float64, 1, vocab))
    
    return Dict(
        :Wemb=>Wemb, :Wpos=>Wpos, :Wq=>Wq, :Wk=>Wk, :Wv=>Wv, :Wo=>Wo,
        :W1=>W1, :b1=>b1, :W2=>W2, :b2=>b2, :Wout=>Wout, :bout=>bout
    )
end

function one_hot_node(idx::Int, vocab::Int)
    v = zeros(Float64, 1, vocab)
    v[1, idx] = 1.0
    return VectNode(v)
end

function embed_sequence(seq::Vector{Int}, Wemb::VectNode, Wpos::VectNode)
    """
    Retourne une matrice (seq_len x d_model) avec embeddings + positional encoding
    """
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

function create_causal_mask(seq_len::Int)
    """
    Crée un masque triangulaire supérieur avec -1e9 pour empêcher l'attention aux positions futures
    """
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

function transpose_node(x::VectNode)
    return VectNode(:transpose, [x], x.value')
end

function masked_scaled_dot_attention(Q::VectNode, K::VectNode, V::VectNode, mask::Matrix{Float64})
    """
    Attention avec masque causal
    Q, K, V: seq_len x d_head
    """
    d = size(Q.value, 2)
    score = (Q * transpose_node(K)) / sqrt(d)  
    
    score_masked = score + VectNode(mask)
    
    weights = softmax(score_masked; dims=2)  
    
    return weights * V 
end

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

function split_heads(x::VectNode, n_heads::Int)
    """
    x: seq_len x d_model
    Retourne une liste de n_heads matrices (seq_len x d_head)
    """
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

function transformer_block(x::VectNode, Wq::VectNode, Wk::VectNode, Wv::VectNode,
                          Wo::VectNode, W1::VectNode, b1::VectNode,
                          W2::VectNode, b2::VectNode, mask::Matrix{Float64}; n_heads=2)
   
    attn_out = multihead_attention(x, Wq, Wk, Wv, Wo, mask; n_heads=n_heads)
    x = layernorm(x + attn_out; eps=1e-5)
    
    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layernorm(x + ff_out; eps=1e-5)
    
    return x
end

function cross_entropy_loss(logits::VectNode, target_idx::Int, vocab::Int)
    
    probs = softmax(logits; dims=2)  
    log_probs = log.(probs .+ VectNode(1e-10))
    
    target_onehot = one_hot_node(target_idx, vocab)
    
    loss = -(target_onehot .* log_probs)
    return sum(loss)
end
function forward_loss_fn(batch_inputs::Vector{Vector{Int}}, batch_targets::Vector{Vector{Int}})
    
    function loss_fn(params_nodes::Flatten)
        Wemb_val, Wpos_val, Wq_val, Wk_val, Wv_val, Wo_val, W1_val, b1_val, W2_val, b2_val, Wout_val, bout_val = params_nodes.components
        
        Wemb = isa(Wemb_val, VectNode) ? Wemb_val : VectNode(Wemb_val)
        Wpos = isa(Wpos_val, VectNode) ? Wpos_val : VectNode(Wpos_val)
        Wq = isa(Wq_val, VectNode) ? Wq_val : VectNode(Wq_val)
        Wk = isa(Wk_val, VectNode) ? Wk_val : VectNode(Wk_val)
        Wv = isa(Wv_val, VectNode) ? Wv_val : VectNode(Wv_val)
        Wo = isa(Wo_val, VectNode) ? Wo_val : VectNode(Wo_val)
        W1 = isa(W1_val, VectNode) ? W1_val : VectNode(W1_val)
        b1 = isa(b1_val, VectNode) ? b1_val : VectNode(b1_val)
        W2 = isa(W2_val, VectNode) ? W2_val : VectNode(W2_val)
        b2 = isa(b2_val, VectNode) ? b2_val : VectNode(b2_val)
        Wout = isa(Wout_val, VectNode) ? Wout_val : VectNode(Wout_val)
        bout = isa(bout_val, VectNode) ? bout_val : VectNode(bout_val)
        
        total_loss = VectNode(0.0)
        B = length(batch_inputs)
        
        for i in 1:B
            seq = batch_inputs[i]
            tgt_seq = batch_targets[i]
            seq_len = length(seq)
            vocab = size(Wout.value, 2)
            
            x = embed_sequence(seq, Wemb, Wpos)  
            
            mask = create_causal_mask(seq_len)
            
            hidden = transformer_block(x, Wq, Wk, Wv, Wo, W1, b1, W2, b2, mask; n_heads=2)
            
            logits = linear_to_vocab(hidden, Wout, bout)  
            
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