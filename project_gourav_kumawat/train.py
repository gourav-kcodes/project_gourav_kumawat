import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from config import (
    CLASS_NAMES, device, CHECKPOINT_PATH,
    learning_rate_phase1, learning_rate_phase2,
    weight_decay, epochs_phase1, epochs_phase2
)


def compute_class_weights(train_df, device):
    train_counts = np.array([
        len(train_df[train_df['label'] == i]) for i in range(len(CLASS_NAMES))
    ])
    class_weights = 1.0 / train_counts
    class_weights = class_weights / class_weights.sum()
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds   = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), correct / total, macro_f1


def train_skin_lesion_model(model, num_epochs, train_loader, loss_fn, optimizer, val_loader=None):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    print(f"Training on {device} for {num_epochs} epochs\n")

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn)

        vl_loss, vl_acc, vl_f1 = 0.0, 0.0, 0.0
        if val_loader is not None:
            vl_loss, vl_acc, vl_f1 = validate_epoch(model, val_loader, loss_fn)
            scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)

        current_lr = optimizer.param_groups[0]['lr']
        saved = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved = '  [saved]'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_val_acc': best_val_acc,
        }, f'checkpoint_epoch_{epoch+1}.pth')

        print(
            f"Ep {epoch+1:02d}/{num_epochs} | "
            f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.3f} | "
            f"VlLoss={vl_loss:.4f}  VlAcc={vl_acc:.3f}  F1={vl_f1:.3f}  "
            f"lr={current_lr:.2e}{saved}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model, history


def run_two_phase_training(model, train_loader, val_loader, train_df):
    class_weights_tensor = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print('=' * 70)
    print(f'  PHASE 1 — Backbone FROZEN  |  Head only for {epochs_phase1} epochs')
    print('=' * 70)
    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate_phase1, weight_decay=weight_decay
    )
    model, history = train_skin_lesion_model(
        model, epochs_phase1, train_loader, criterion, optimizer1, val_loader
    )

    for param in model.parameters():
        param.requires_grad = True

    print('\n' + '=' * 70)
    print(f'  PHASE 2 — ALL LAYERS UNFROZEN  |  Fine-tune for {epochs_phase2} epochs')
    print('=' * 70)
    optimizer2 = optim.Adam(
        model.parameters(), lr=learning_rate_phase2, weight_decay=weight_decay
    )
    model, history2 = train_skin_lesion_model(
        model, epochs_phase2, train_loader, criterion, optimizer2, val_loader
    )

    for key in history:
        history[key].extend(history2[key])

    return model, history
