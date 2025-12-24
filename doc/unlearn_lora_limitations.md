# UNLEARN LoRA Filtering Notes

This patch normalizes parameter names by trimming a leading `model.` prefix so that LoRA adapters loaded through HuggingFace/PEFT survive `_param_filter` when wrapped by `AdapterModel`. The change is tailored to the LLaMA LoRA stack that emits names like `model.base_model.*` from `named_parameters()` while the serialized state dict drops the outer prefix.

## Known limitations
- **Model-specific assumption:** The normalization only strips the first occurrence of `model.`. Architectures whose adapters introduce different prefixes (for example Qwen PEFT wrappers that prepend `base.` or custom nesting) will still mismatch and get filtered out. Extending support requires auditing each model family and adding the appropriate normalization rules.
- **Single-level prefix removal:** If a backend adds multiple nested wrappers and removes more than one token when materializing the state dict, the current trimming will not recover those parameter names.
- **Non-LoRA adapters:** The logic has only been validated for LoRA matrices emitted by PEFT. Other adapter types might rename their tensors differently; they have not been exercised.

Please revisit this helper before enabling UNLEARN for additional model families.
