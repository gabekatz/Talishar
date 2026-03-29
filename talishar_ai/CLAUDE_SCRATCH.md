# CLAUDE_SCRATCH.md — Living Ledger

## Session: 2026-03-29 — MLX LoRA Training Completed

### What Happened
1. **MLX LoRA fine-tuning completed successfully** (1000 iterations, ~2 hours on Apple Silicon)
   - Model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
   - Train loss: 2.518 → 0.078 (32x improvement)
   - Val loss: 2.518 → 0.079 (best: 0.077 at iter 900)
   - No overfitting — val tracks train closely
   - Peak memory: 26.8 GB
   - Adapters saved at: `distill_model/adapters/` (44MB each, 5 checkpoints)

2. **Test accuracy: 70% on 20 test samples**
   - Model produces well-structured JSON (action_index, reasoning, phase_strategy, confidence)
   - Misses are typically close alternatives, not wild errors
   - Understands pitching, go-again chains, opponent hand state

3. **Committed and pushed** all changes to `feature/ai-training`:
   - `.gitignore` (excludes checkpoints, __pycache__, training artifacts)
   - `scripts/distill_local.py` — MLX LoRA pipeline (prepare/train/test/fuse)
   - `scripts/diagnose_game.py` — Model behavior diagnostic tool
   - `scripts/fix_metadata_values.py` — Card metadata enrichment (464 on_hit, 583 effect, 141 block)
   - `env.py` — combatChain null fix, handCount tracking fix
   - `card_metadata.json` — Enriched with on_hit_value, effect_value, block_willingness
   - `training/async_trainer.py` — Better error tracing
   - `hand_evaluator.py` — Hand evaluation utility
   - `llm_demos/data.jsonl` — LLM demo data
   - Removed all accidentally committed `__pycache__/` files

### Key Fixes Made
- `distill_local.py`: Fixed mlx_lm API changes (CLI args → JSON config, `temp` → `sampler` function)
- `env.py`: `state.get("combatChain", {})` → `state.get("combatChain") or {}` (PHP null handling)
- `env.py`: Opponent hand size uses `handCount` field instead of `hand` array

### Pending Next Steps
1. **Fuse the LoRA adapters** into a standalone model: `uv run python -m scripts.distill_local fuse`
2. **Wire local model into play_llm.py** as a drop-in replacement for Claude API
3. **Generate more BC data** using the local model (free, unlimited games)
4. **Increase equip_penalty_scale** in env.py (currently 0.02, too weak vs ±1.0 terminal reward)
5. **Run PPO training** with BC-pretrained weights using fixed metadata and env
6. **Consider**: Run another 25 Claude games for higher-quality distillation data to improve the 70% accuracy

### Architecture Notes
- LoRA adapters at `distill_model/adapters/adapters.safetensors` (44MB)
- Training data at `distill_data/{train,valid,test}.jsonl` (6171/342/342 records)
- Source distillation data at `llm_demos/distill.jsonl` (6855 total, 25 games)
- Checkpoints in `distill_model/adapters/0000{200,400,600,800,1000}_adapters.safetensors`
