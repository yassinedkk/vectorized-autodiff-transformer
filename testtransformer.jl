using Random
using Printf
using Statistics
using StatsBase
using HTTP

# Includes
include("flatten.jl")
include("forward.jl")
include("models.jl")
include("reverse_vectorized.jl")
include("transformer.jl")

using .Forward
using .VectReverse
using .Transformer

println("="^70)
println("TEST TRANSFORMER COMPLET")
println("="^70)

# CHOIX DU TYPE DE TEST
println("\nChose the type :")
println("  1. tets (Copy, Reverse, Shift, Repeat, Markov)")
println("  2. Shakespeare (generation of text)")
print("\nYour choice [1-2] : ")

test_type = 1
try
    global test_type
    input_str = readline()
    if !isempty(input_str)
        test_type = parse(Int, input_str)
    end
catch
    println("test 1")
end

# globale variables 
vocab = 10
d_model = 32
d_ff = 128
seq_len = 8
max_seq_len = 10
batch_size = 8
learning_rate = 0.01
max_grad_norm = 30.0
num_epochs = 1500
n_layer = 1
n_head = 2

rng = MersenneTwister(1234)

if test_type == 2
    println("\nMODE SHAKESPEARE")
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    println("\nTéléchargement du dataset Tiny Shakespeare...")
    if !isfile("shakespeare.txt")
        try
            response = HTTP.get(shakespeare_url)
            shakespeare_text = String(response.body)
            open("shakespeare.txt", "w") do f
                write(f, shakespeare_text)
            end
            println("Dataset téléchargé.")
        catch e
            error("Erreur de téléchargement du dataset: $e")
        end
    else
        shakespeare_text = read("shakespeare.txt", String)
        println("Dataset chargé depuis le cache.")
    end

    # char-level vocab
    chars = sort(unique(collect(shakespeare_text)))
    vocab = length(chars)
    char_to_idx = Dict(c => i for (i, c) in enumerate(chars))
    idx_to_char = Dict(i => c for (i, c) in enumerate(chars))

    data = [char_to_idx[c] for c in shakespeare_text]
    n_train = Int(floor(0.9 * length(data)))
    train_data = data[1:n_train]
    val_data = data[n_train+1:end]

    # hyper parameters for Shakespeare
    d_model = 128
    d_ff = 256
    seq_len = 64
    max_seq_len = 64
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = 10000
    n_layer = 2
    n_head = 4

    function get_batch_shakespeare(split, train_data, val_data, seq_len, batch_size, rng)
        data_source = split == :train ? train_data : val_data
        inputs = Vector{Vector{Int}}(undef, batch_size)
        targets = Vector{Vector{Int}}(undef, batch_size)
        N = length(data_source)
        for i in 1:batch_size
            ix = rand(rng, 1:(N - seq_len - 1))
            inputs[i] = data_source[ix:ix+seq_len-1]
            targets[i] = data_source[ix+1:ix+seq_len]
        end
        return inputs, targets
    end

    generate_batch = (bs, sl, v, r) -> get_batch_shakespeare(:train, train_data, val_data, sl, bs, r)
    task_name = "Shakespeare"

else
    println("\nMODE")

    function generate_copy_task(batch_size, seq_len, vocab, rng)
        inputs = [rand(rng, 1:vocab, seq_len) for _ in 1:batch_size]
        targets = copy.(inputs)
        return inputs, targets
    end

    function generate_reverse_task(batch_size, seq_len, vocab, rng)
        inputs = [rand(rng, 1:vocab, seq_len) for _ in 1:batch_size]
        targets = [reverse(seq) for seq in inputs]
        return inputs, targets
    end

    function generate_shift_task(batch_size, seq_len, vocab, rng; k=1)
        inputs = [rand(rng, 1:vocab, seq_len) for _ in 1:batch_size]
        targets = [[mod1(x + k, vocab) for x in seq] for seq in inputs]
        return inputs, targets
    end

    function generate_repeat_pattern(batch_size, seq_len, vocab, rng; pattern_len=2)
        inputs = Vector{Vector{Int}}(undef, batch_size)
        targets = Vector{Vector{Int}}(undef, batch_size)
        for i in 1:batch_size
            pattern = rand(rng, 1:vocab, pattern_len)
            full_seq = repeat(pattern, div(seq_len, pattern_len) + 2)
            inputs[i] = full_seq[1:seq_len]
            targets[i] = full_seq[2:seq_len+1]
        end
        return inputs, targets
    end

    function make_random_markov(vocab, rng; smooth=1e-3)
        P = rand(rng, vocab, vocab)
        for i in 1:vocab
            P[i, :] .+= smooth
            P[i, :] ./= sum(P[i, :])
        end
        return P
    end

    function generate_markov_batch(batch_size, seq_len, P, rng)
        vocab = size(P, 1)
        inputs = Vector{Vector{Int}}(undef, batch_size)
        targets = Vector{Vector{Int}}(undef, batch_size)
        for b in 1:batch_size
            seq = Vector{Int}(undef, seq_len+1)
            seq[1] = rand(rng, 1:vocab)
            for t in 2:seq_len+1
                cur = seq[t-1]
                seq[t] = wsample(rng, 1:vocab, view(P, cur, :))
            end
            inputs[b] = seq[1:seq_len]
            targets[b] = seq[2:seq_len+1]
        end
        return inputs, targets
    end

    println("\nChoose a task:")
    println("  1. Copy")
    println("  2. Reverse")
    println("  3. Shift (+1)")
    println("  4. Repeat Pattern")
    println("  5. Markov (random chain)")
    print("\nYour choice [1-5] : ")

    task_choice = 1
    try
        global task_choice
        task_input = readline()
        if !isempty(task_input)
            task_choice = parse(Int, task_input)
        end
    catch
        println("Using task 1 (Copy)")
    end

    P_markov = make_random_markov(vocab, rng)

    generate_batch = if task_choice == 2
        generate_reverse_task
    elseif task_choice == 3
        generate_shift_task
    elseif task_choice == 4
        (bs, sl, v, r) -> generate_repeat_pattern(bs, sl, v, r; pattern_len=2)
    elseif task_choice == 5
        (bs, sl, v, r) -> generate_markov_batch(bs, sl, P_markov, r)
    else
        generate_copy_task
    end

    task_name = ["Copy", "Reverse", "Shift", "Repeat Pattern", "Markov"][task_choice]
end

println("\nTask: $task_name")
println("  Vocab       : $vocab")
println("  d_model     : $d_model")
println("  Seq length  : $seq_len")
println("  Batch size  : $batch_size")
println("  LR          : $learning_rate")
println("  Epochs      : $num_epochs")
println("  Layers      : $n_layer")
println("  Heads       : $n_head")


function clip_gradients!(g::Flatten, max_norm::Float64)
    total_norm = sqrt(sum(sum(comp.^2) for comp in g.components))
    if total_norm > max_norm
        scale = max_norm / total_norm
        for i in eachindex(g.components)
            g.components[i] .*= scale
        end
        return total_norm, true
    end
    return total_norm, false
end

function check_gradient_health(g::Flatten)
    has_nan = any(any(isnan.(comp)) for comp in g.components)
    has_inf = any(any(isinf.(comp)) for comp in g.components)
    total_norm = sqrt(sum(sum(comp.^2) for comp in g.components))
    return !has_nan && !has_inf && total_norm < 1e6
end

# INITIALISATION
println("\nModel initialization")
pdict = Transformer.init_transformer_params(vocab, d_model, d_ff, max_seq_len; n_layers=n_layer, rng=rng)
params_flat = Transformer.flatten_params(pdict)
total_params = sum(length(comp) for comp in params_flat.components)
println("Total parameters: $total_params")

# Test forward/backward
println("\nTest forward/backward...")
test_inputs, test_targets = generate_batch(2, seq_len, vocab, rng)
loss_fn = Transformer.forward_loss_fn(test_inputs, test_targets; n_heads=n_head)
test_loss_node = loss_fn(params_flat)
@printf("Initial loss: %.6f\n", test_loss_node.value)
test_g = VectReverse.gradient(loss_fn, params_flat)
@printf("Gradient computed successfully (test).\n")

# PREPARE BENCHMARK ARRAYS
losses = Float64[]
grad_norms = Float64[]
grad_times = Float64[]
epoch_times = Float64[]
n_clips = 0
n_bad_grad = 0

display_interval = test_type == 2 ? 500 : 100

println("\nStarting training...")

for epoch in 1:num_epochs
    t_epoch_start = time()

    # Generate batch and loss function
    inputs, targets = generate_batch(batch_size, seq_len, vocab, rng)
    loss_fn = Transformer.forward_loss_fn(inputs, targets; n_heads=n_head)

    # Forward
    loss_node = loss_fn(params_flat)
    loss_val = loss_node.value
    push!(losses, loss_val)

    # Backward with timing
    t_back_start = time()
    g = VectReverse.gradient(loss_fn, params_flat)
    t_back_end = time()
    t_backward = t_back_end - t_back_start
    push!(grad_times, t_backward)

    # Gradient health check
    if !check_gradient_health(g)
        n_bad_grad += 1
        println("Warning: invalid gradient detected at epoch $epoch (NaN/Inf or too large); skipping update")
        push!(grad_norms, NaN)
        push!(epoch_times, time() - t_epoch_start)
        continue
    end

    # Compute grad norm + clipping
    grad_norm, clipped = clip_gradients!(g, max_grad_norm)
    push!(grad_norms, grad_norm)
    if clipped
        #n_clips += 1
    end

    # Update parameters (SGD)
    for i in eachindex(params_flat.components)
        params_flat.components[i] .-= learning_rate .* g.components[i]
    end

    t_epoch_end = time()
    push!(epoch_times, t_epoch_end - t_epoch_start)

    # Display
    if epoch % display_interval == 0 || epoch == 1
        @printf("Epoch %5d | Loss = %.6f | GradNorm = %.3f | Backward = %.4fs | Epoch = %.4fs | Clipped = %s\n",
                epoch, loss_val, grad_norm, t_backward, epoch_times[end], clipped ? "YES" : "NO")
    end

    # Safety stop
    if !isfinite(loss_val)
        println("Loss became NaN/Inf at epoch $epoch; stopping.")
        break
    end
end

println("\nTraining finished.\n")

# BENCHMARK REPORT
println(" BENCHMARK REPORT ")
@printf("Initial loss: %.6f\n", losses[1])
@printf("Final loss:   %.6f\n", losses[end])
reduction = 100 * (1 - losses[end] / losses[1])
@printf("Loss reduction: %.2f%%\n", reduction)

# Times
mean_epoch_time = mean(epoch_times)
std_epoch_time = std(epoch_times)
mean_backward = mean(grad_times)
std_backward = std(grad_times)
@printf("Mean epoch time     : %.6f s (std %.6f)\n", mean_epoch_time, std_epoch_time)
@printf("Mean backward time  : %.6f s (std %.6f)\n", mean_backward, std_backward)
@printf("Ratio backward/epoch: %.4f\n", mean_backward / mean_epoch_time)

# Grad norms
valid_grad_norms = filter(!isnan, grad_norms)
if !isempty(valid_grad_norms)
    @printf("Grad norm mean : %.6f\n", mean(valid_grad_norms))
    @printf("Grad norm std  : %.6f\n", std(valid_grad_norms))
    @printf("Grad norm max  : %.6f\n", maximum(valid_grad_norms))
    @printf("Grad norm min  : %.6f\n", minimum(valid_grad_norms))
else
    println("No valid gradient norms recorded.")
end

@printf("Number of clipping events: %d / %d (%.2f%%)\n", n_clips, length(losses), 100 * n_clips / max(1, length(losses)))
@printf("Number of invalid gradients skipped: %d\n", n_bad_grad)

# LOSS PLOT
function ascii_plot(data::Vector{Float64}, height::Int=10, width::Int=60)
    if isempty(data)
        println("No data to plot.")
        return
    end
    minv, maxv = minimum(data), maximum(data)
    rangev = maxv - minv
    if rangev < 1e-12
        println("Loss constant at $(round(minv, digits=6))")
        return
    end
    step = max(1, Int(div(length(data), width)))
    sampled = [data[i] for i in 1:step:length(data)]
    for h in height:-1:1
        threshold = minv + (h / height) * rangev
        line = ""
        for val in sampled
            line *= (val >= threshold) ? "█" : " "
        end
        @printf("%7.3f |%s\n", threshold, line)
    end
    println(" " ^ 8 * "+" * "-" ^ length(sampled))
end

println("LOSS TREND (ASCII)")
ascii_plot(losses)

# TESTS / GENERATION
if test_type == 2
    println("\nGENERATION (Shakespeare)")

    function generate_shakespeare(params_flat, start_text, max_tokens, seq_len, char_to_idx, idx_to_char, n_head; temp=0.8)
        idx = [char_to_idx[c] for c in collect(start_text)]
        p = Transformer.unflatten_params(params_flat; n_layers=n_layer)

        for _ in 1:max_tokens
            idx_cond = idx[max(1, length(idx) - seq_len + 1):end]
            x = Transformer.embed_sequence(idx_cond, p[:Wemb], p[:Wpos])
            mask = Transformer.create_causal_mask(length(idx_cond))
            # forward through layers
            for layer in 1:n_layer
                x = Transformer.transformer_block(
                    x,
                    p[Symbol("Wq_$layer")], p[Symbol("Wk_$layer")],
                    p[Symbol("Wv_$layer")], p[Symbol("Wo_$layer")],
                    p[Symbol("W1_$layer")], p[Symbol("b1_$layer")],
                    p[Symbol("W2_$layer")], p[Symbol("b2_$layer")],
                    mask; n_heads=n_head
                )
            end
            logits = Transformer.linear_to_vocab(x, p[:Wout], p[:bout])
            logits_last = logits.value[end:end, :]
            logits_scaled = logits_last ./ temp
            logits_exp = exp.(logits_scaled .- maximum(logits_scaled))
            probs = logits_exp ./ sum(logits_exp)
            idx_next = sample(1:length(probs[1,:]), Weights(probs[1,:]))
            push!(idx, idx_next)
        end

        return String([idx_to_char[i] for i in idx])
    end

    for prompt in ["\n", "ROMEO:"]
        println("\nPrompt: ", repr(prompt))
        generated = generate_shakespeare(params_flat, prompt, 200, seq_len, char_to_idx, idx_to_char, n_head; temp=0.8)
        println(generated)
    end

    if losses[end] < 1.5
        println("\nFinal evaluation: very coherent text")
    elseif losses[end] < 2.0
        println("\nFinal evaluation: reasonable structure")
    else
        println("\nFinal evaluation: needs more training or capacity")
    end

else
    println("\nPATTERNS ACCURACY TESTS")

    function predict_next(seq, params_flat, n_head)
        p = Transformer.unflatten_params(params_flat)
        x = Transformer.embed_sequence(seq, p[:Wemb], p[:Wpos])
        mask = Transformer.create_causal_mask(length(seq))

        for layer in 1:n_layer
            x = Transformer.transformer_block(
                x,
                p[Symbol("Wq_$layer")], p[Symbol("Wk_$layer")],
                p[Symbol("Wv_$layer")], p[Symbol("Wo_$layer")],
                p[Symbol("W1_$layer")], p[Symbol("b1_$layer")],
                p[Symbol("W2_$layer")], p[Symbol("b2_$layer")],
                mask; n_heads=n_head
            )
        end

        logits = Transformer.linear_to_vocab(x, p[:Wout], p[:bout])
        return argmax(logits.value[end, :])
    end

    test_inputs, test_targets = generate_batch(5, seq_len, vocab, rng)

    global total_correct = 0
    global total_tokens = 0

    for i in 1:5
        input_seq = test_inputs[i]
        target_seq = test_targets[i]

        pred_seq = [predict_next(input_seq[1:pos], params_flat, n_head) for pos in 1:seq_len]

        correct = sum(pred_seq .== target_seq)

        global total_correct += correct
        global total_tokens += seq_len

        match = pred_seq == target_seq ? "OK" : "FAIL"
        println("$i. Input: $input_seq -> Target: $target_seq")
        println("   Predicted: $pred_seq ($correct/$seq_len) $match")
    end

    accuracy = 100 * total_correct / total_tokens
    println("\nAccuracy: $(round(accuracy, digits=1))%")
end

println("\nTest completed.")
