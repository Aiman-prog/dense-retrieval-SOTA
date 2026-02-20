"""
Temperature-scaled contrastive loss for dense retrieval.

Fixes the bug where Tevatron's GradCache trainer uses SimpleContrastiveLoss
which doesn't apply temperature scaling, resulting in loss values of 80+
instead of the expected 2-10 range.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor


class TemperatureScaledContrastiveLoss:
    """SimpleContrastiveLoss with temperature scaling support.
    
    This is a drop-in replacement for tevatron.retriever.gc_trainer.SimpleContrastiveLoss
    that applies temperature scaling to logits before computing cross-entropy loss.
    
    Args:
        temperature: Temperature parameter for contrastive learning (default: 0.02)
                    Lower values create sharper softmax distributions.
    """
    
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        """Compute temperature-scaled contrastive loss.
        
        Args:
            x: Query representations [batch_size, hidden_dim]
            y: Passage representations [num_passages, hidden_dim]
            target: Target indices (optional, auto-generated if None)
            reduction: Loss reduction method ('mean' or 'sum')
            
        Returns:
            Scalar loss value
        """
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, 
                device=x.device, dtype=torch.long
            )
        
        # Compute similarity scores
        logits = torch.matmul(x, y.transpose(0, 1))
        
        # ✅ KEY FIX: Apply temperature scaling
        # This was missing in the original SimpleContrastiveLoss
        logits = logits / self.temperature
        
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedTemperatureScaledContrastiveLoss(TemperatureScaledContrastiveLoss):
    """Distributed version with temperature scaling.
    
    This is a drop-in replacement for tevatron.retriever.gc_trainer.DistributedContrastiveLoss
    """
    
    def __init__(self, temperature: float = 0.02, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature=temperature)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
