"""Training components for InstructGS."""

from .round_trainer import RoundBasedTrainer, create_trainer

__all__ = ['RoundBasedTrainer', 'create_trainer']