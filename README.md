# Full Fine-tuning vs PEFT: Knowledge Preservation Analysis

This repository demonstrates a comparative analysis between **Full Fine-tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** methods, specifically examining how both approaches preserve pre-trained knowledge while adapting to new tasks.

## Overview

The notebook `full_fine_tuning_vs_peft.ipynb` implements both fine-tuning strategies on a T5-small model, training it for Text-to-SQL generation while evaluating how well each method preserves the model's original capabilities (e.g., translation tasks).

## Knowledge Preservation Strategies

### 1. Full Fine-tuning Approach

**How it preserves knowledge:**

- Updates all model parameters during training
- Relies on careful hyperparameter tuning to prevent catastrophic forgetting
- Uses techniques like:
  - Low learning rates to make gradual updates
  - Early stopping to prevent overtraining
  - Regularization methods to maintain original capabilities

**Trade-offs:**

- ✅ Can achieve high task-specific performance
- ✅ Full model adaptability
- ❌ Risk of catastrophic forgetting
- ❌ Computationally expensive (all parameters updated)
- ❌ Requires more storage for each fine-tuned variant

### 2. PEFT (LoRA) Approach

**How it preserves knowledge:**

- **Keeps original weights frozen**: Base model parameters remain unchanged
- **Adds low-rank adaptation layers**: Only trains small additional matrices
- **Selective module targeting**: Applies LoRA to specific attention components (q, v, o modules)
- **Rank decomposition**: Uses rank-16 matrices to capture task-specific adaptations

**Configuration used:**

```python
peft_config = LoraConfig(
    lora_alpha=16,        # Scaling factor for LoRA updates
    lora_dropout=0.5,     # Prevents overfitting in LoRA layers
    r=16,                 # Low rank for efficient adaptation
    bias="none",          # No additional bias terms
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v", "o"],  # Attention components only
    modules_to_save=["lm_head"],     # Save final prediction layer
)
```

**Knowledge preservation mechanism:**

- Original pre-trained weights remain intact (knowledge base preserved)
- New task knowledge is learned through small adapter layers
- Both capabilities can coexist without interference
- Efficient parameter usage (~1-2% of original parameters)

## Experimental Results

### Knowledge Preservation Test

Both approaches are tested on:

1. **Original task**: English to French translation
2. **New task**: Text-to-SQL generation

### Key Findings

**Parameter Efficiency:**

- Full fine-tuning: Updates 100% of model parameters
- PEFT (LoRA): Updates only ~1-2% of parameters while maintaining performance

**Knowledge Retention:**

- **Full fine-tuning**: May lose translation capabilities if not carefully tuned
- **PEFT**: Maintains translation capabilities while gaining SQL generation skills

**Performance Metrics (ROUGE scores):**

- Both methods show improvement on the target task
- PEFT demonstrates better knowledge preservation
- Full fine-tuning may achieve slightly higher task-specific performance

## Why PEFT Preserves Knowledge Better

1. **Architectural Separation**: Original knowledge in base weights, new knowledge in adapters
2. **Minimal Interference**: Low-rank updates don't overwrite existing representations
3. **Modular Design**: Can add/remove task-specific adapters without affecting base model
4. **Gradient Isolation**: Original parameters don't receive gradient updates

## Use Cases

**Choose Full Fine-tuning when:**

- Maximum task performance is critical
- Computational resources are abundant
- Single-task deployment scenario
- Domain shift is substantial

**Choose PEFT when:**

- Need to preserve multiple capabilities
- Resource-constrained environments
- Multi-task model requirements
- Want to avoid catastrophic forgetting
- Need to switch between different adapted versions

## Repository Structure

- `full_fine_tuning_vs_peft.ipynb`: Main implementation and comparison
- `README.md`: This documentation
- Model outputs demonstrate both approaches maintaining translation while learning SQL generation

## Key Takeaways

1. **PEFT offers superior knowledge preservation** by design through parameter isolation
2. **Full fine-tuning requires careful tuning** to avoid forgetting but can achieve higher task-specific performance
3. **Both methods can coexist** - use PEFT for most scenarios, full fine-tuning when maximum performance is needed
4. **PEFT is more practical** for real-world deployments due to efficiency and modularity

This analysis provides insights into choosing the right fine-tuning strategy based on your specific requirements for knowledge preservation and task performance.