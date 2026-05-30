## Ablation Study Results

Evaluated on 20 HumanEval problems. Model: qwen2.5-coder:0.5b. Hardware: Intel Iris Xe, 8GB RAM, CPU-only inference.

| Mode | Avg BLEU | pass@1 | Exact Match | Avg Latency |
|---|---|---|---|---|
| baseline | 0.5809 | 85.0% | 20.0% | 5937ms |
| naive | 0.6717 | 90.0% | 40.0% | 8214ms |
| localcodepilot | 0.7105 | 95.0% | 35.0% | 8428ms |

**LocalCodePilot vs Baseline:** BLEU +22.3% | pass@1 +10.0 percentage points