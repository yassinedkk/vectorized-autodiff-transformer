# Vectorized automatic differentiation and transformers in Julia

Portfolio project for **LINMA2472 — Algorithms in Data Science**. The work implements vectorized reverse-mode automatic differentiation, forward-over-reverse Hessian–vector products, optimization experiments, and a small character-level transformer trained on Shakespeare text.

## Highlights

- Tensor-aware reverse-mode AD with broadcasting and matrix operations
- Hessian–vector products without explicitly materializing the full Hessian
- Gradient-descent and Newton-CG experiments
- Character-level transformer experiments on Tiny Shakespeare
- Reproducible benchmark and result slides

## Experimental results

The measurements below come from the submitted [results presentation](https://github.com/yassinedkk/LDAT2M/blob/main/portfolio/vectorized-autodiff-transformer/results_presentation.pptx). They were not rerun during portfolio packaging.

### Vectorized reverse-mode gradients

| Task | Configuration | Parameters | Forward time | Forward memory | Reverse time | Reverse memory | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regression | n=32, d=16, h=16 | 272 | 3.12e-2 s | 35.21 MB | 2.15e-4 s | 0.06 MB | 145× |
| Regression | n=64, d=32, h=32 | 1,056 | 5.44e-1 s | 521.29 MB | 2.66e-4 s | 0.21 MB | 2,041× |
| Classification | n=32, d=16, h=16 | 304 | 5.26e-1 s | 407.35 MB | 3.26e-4 s | 0.11 MB | 1,611× |

For these configurations, vectorized reverse mode reduced gradient runtime by **145× to 2,041×** and used much less memory than scalar forward mode.

### Forward-over-reverse Hessian–vector products

| Task | Configuration | Parameters | Forward time | RoF time | Speedup | Forward memory | RoF memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regression | n=32, d=16, h=16 | 272 | 0.1244 s | 0.00044 s | 285× | 84.53 MB | 0.17 MB |
| Regression | n=64, d=32, h=32 | 1,056 | 1.7951 s | 0.00068 s | 2,629× | 1,267.49 MB | 0.62 MB |
| Classification | n=64, d=32, h=32 | 1,120 | 35.4565 s | 0.00099 s | 35,742× | 26,475.79 MB | 0.88 MB |

The forward-over-reverse implementation produced the largest reported gain on the classification benchmark, with a **35,742×** speedup and memory use falling from **26,475.79 MB to 0.88 MB**.

### Optimization after five iterations

| Task | Method | Time | Final loss |
| --- | --- | ---: | ---: |
| Regression (n=64, d=16, h=16) | Gradient descent | 1.18 s | 1.289 |
| Regression (n=64, d=16, h=16) | Newton-CG | 2.81 s | 0.679 |
| Classification (n=64, d=16, h=16) | Gradient descent | 6.14 s | 1.3071 |
| Classification (n=64, d=16, h=16) | Newton-CG | 6.61 s | 1.2650 |

Newton-CG achieved a lower final loss in both experiments. The improvement was larger for regression, while the classification runtimes were close.

![Computation graph](https://github.com/yassinedkk/LDAT2M/blob/main/portfolio/vectorized-autodiff-transformer/dag.png)

## Run

Requires Julia.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmark.jl
julia --project=. testtransformer.jl
```

`testtransformer.jl` offers basic transformer tests and Shakespeare text generation. The dataset is included as `shakespeare.txt`.

## Main files

- `reverse_vectorized.jl`: vectorized reverse-mode AD and Hessian–vector products
- `benchmark.jl`: gradient, HVP, and optimizer benchmarks
- `transformer.jl`: multi-layer transformer implementation
- `transformer_single_layer.jl`: compact single-layer variant
- `testtransformer.jl`: tests and character-generation experiment
- `results_presentation.pptx`: experimental results and interpretation
- `shakespeare.txt`: Tiny Shakespeare corpus used by the experiment

## Authors

- Yassine Zeamari
- Gauthier Viseur
- Mehdi Mannane

## Course attribution and license

This project was completed from the LINMA2472 course assignment and supporting material by **Benoît Legat**. Course-derived code and this published project are provided under the MIT License; see [LICENSE](LICENSE).

Assignment statement: [LINMA2472 HomeworkAD](https://github.com/blegat/LINMA2472/blob/main/HomeworkAD/README.md).

The Shakespeare corpus is redistributed for this educational experiment. See [DATA_NOTICE.md](DATA_NOTICE.md) for its source and attribution.

## Packaging note

The portfolio copy removes redundant lab/reference files and fixes filename casing plus missing dependency declarations for portability. A Julia runtime was unavailable in the packaging environment, so the code received static checks but was not executed there.


> **Project archive:** Large binary artifacts are available in the [original portfolio folder](https://github.com/yassinedkk/LDAT2M/tree/main/portfolio/vectorized-autodiff-transformer).
