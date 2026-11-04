import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction="none",
    )
    mask = attention_mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def music_loss(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    notes: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    attribute_mask = attention_mask
    if eos_token_id is not None:
        attribute_mask = attention_mask * (notes[..., 0] != eos_token_id)
    losses = {
        "start": masked_cross_entropy(logits[0], notes[..., 0], attention_mask),
        "pitch": masked_cross_entropy(logits[1], notes[..., 1], attribute_mask),
        "duration": masked_cross_entropy(
            logits[2],
            notes[..., 2],
            attribute_mask,
        ),
    }
    total = sum(losses.values()) / len(losses)
    return total, losses



