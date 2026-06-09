"""Reinforcement-learning trainer for the state translator.

A Gymnasium environment (``TranslationEnv``) turns a BBOB run with optimizer
switches into an MDP; a custom PPO loop (``ppo``) trains the set-equivariant
``TranslatorPair`` as the policy (``policy.ActorCritic``) to maximize true
downstream optimization improvement plus a cycle-consistency penalty.
"""
