"""
talishar_ai — Reinforcement learning agent for Flesh and Blood (Talishar engine).

Package layout:
  game_manager.py   HTTP wrappers for the PHP engine
  features.py       State featurisation (JSON → numpy)
  env.py            Gymnasium-compatible environment
  models/           Neural network (Actor-Critic)
  training/         PPO algorithm, rollout buffer, training loop
  scripts/          CLI entry-points (train, evaluate)
"""
