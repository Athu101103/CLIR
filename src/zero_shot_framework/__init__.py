# Zero-shot Framework for Anveshana CLIR

from .base_zs import BaseZSModel
from .xlm_roberta_zs import XLMRoBERTaZSModel
from .contriever_zs import ContrieverZSModel
from .e5_zs import MultilingualE5ZSModel
from .zs_runner import ZSRunner

# Optional ColBERT import (guarded to avoid heavy deps/TensorFlow import issues)
import os as _os
try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

_ENABLE_COLBERT = (_torch is not None and getattr(_torch, 'cuda', None) and _torch.cuda.is_available() and _os.getenv('ENABLE_COLBERT') == '1')
if _ENABLE_COLBERT:
    try:
        from .colbert_zs import ColBERTZSModel  # type: ignore
    except Exception:
        ColBERTZSModel = None  # type: ignore
else:
    ColBERTZSModel = None  # type: ignore

__all__ = [
    'BaseZSModel',
    'XLMRoBERTaZSModel',
    'ContrieverZSModel',
    'MultilingualE5ZSModel',
    'ZSRunner',
]

# Export ColBERT if available
if ColBERTZSModel is not None:
    __all__.append('ColBERTZSModel')
