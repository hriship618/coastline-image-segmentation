"""The smallest useful PyTorch segmentation training primitives."""

from __future__ import annotations

from pathlib import Path

from coastlearn.metrics import SegmentationConfusionMatrix


def build_cross_entropy_loss(ignore_index: int = 255, class_weights=None):
    """Create pixel-wise cross-entropy for integer segmentation masks."""
    import torch
    import torch.nn as nn

    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


def validate_segmentation_shapes(logits, masks) -> None:
    """Fail early when model outputs cannot correspond to target pixels."""
    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W], got {tuple(masks.shape)}")
    if logits.shape[0] != masks.shape[0]:
        raise ValueError("Logits and masks have different batch sizes")
    if logits.shape[-2:] != masks.shape[-2:]:
        raise ValueError("Logits and masks have different spatial sizes")


def train_one_batch(model, images, masks, optimizer, loss_function) -> dict[str, object]:
    """Run forward pass, loss, backpropagation, and one optimizer update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    validate_segmentation_shapes(logits, masks)
    loss = loss_function(logits, masks.long())

    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "logits_shape": tuple(logits.shape),
    }


def build_finetuning_optimizer(
    model,
    backbone_learning_rate: float = 1e-5,
    head_learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """Use conservative backbone updates and faster segmentation-head updates."""
    import torch

    if backbone_learning_rate <= 0 or head_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")

    if hasattr(model, "backbone"):
        backbone = model.backbone
    elif hasattr(model, "encoder"):
        backbone = model.encoder
    else:
        raise ValueError("Model must expose a backbone or encoder")

    backbone_parameters = list(backbone.parameters())
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_parameter_ids
    ]
    if not head_parameters:
        raise ValueError("No segmentation-head parameters were found")

    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": head_learning_rate,
                "name": "head",
            },
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(model, dataloader, optimizer, loss_function, device) -> float:
    """Train on every batch once and return mean loss per image."""
    model.train()
    total_loss = 0.0
    total_images = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        result = train_one_batch(model, images, masks, optimizer, loss_function)
        batch_size = images.shape[0]
        total_loss += result["loss"] * batch_size
        total_images += batch_size

    if total_images == 0:
        raise ValueError("Training dataloader is empty")
    return total_loss / total_images


def evaluate(model, dataloader, loss_function, device, num_classes: int = 2) -> dict:
    """Evaluate without gradients and calculate loss and IoU."""
    import torch

    model.eval()
    confusion = SegmentationConfusionMatrix(num_classes=num_classes)
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            validate_segmentation_shapes(logits, masks)
            loss = loss_function(logits, masks.long())
            predictions = logits.argmax(dim=1)

            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size
            confusion.update(predictions.cpu().numpy(), masks.cpu().numpy())

    if total_images == 0:
        raise ValueError("Evaluation dataloader is empty")
    metrics = confusion.summary()
    metrics["loss"] = total_loss / total_images
    return metrics


def save_checkpoint(path, model, optimizer, epoch: int, validation_metrics: dict) -> None:
    """Save enough state to reproduce or resume the best model."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
        },
        path,
    )


def fit(
    model,
    train_loader,
    validation_loader,
    optimizer,
    loss_function,
    device,
    epochs: int,
    checkpoint_path,
    patience: int = 5,
) -> list[dict]:
    """Train, validate, retain the best mean-IoU checkpoint, and stop early."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")

    history = []
    best_mean_iou = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        validation_metrics = evaluate(
            model, validation_loader, loss_function, device
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_mean_iou": validation_metrics["mean_iou"],
            "validation_class_iou": validation_metrics["class_iou"],
        }
        history.append(epoch_result)
        print(epoch_result)

        if validation_metrics["mean_iou"] > best_mean_iou:
            best_mean_iou = validation_metrics["mean_iou"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_metrics,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return history
