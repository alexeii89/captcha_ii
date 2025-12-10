import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import CRNN, decode_predictions
from dataset_loader import get_data_loaders
import matplotlib.pyplot as plt


def train_epoch(model, train_loader, criterion, optimizer, device, dataset):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Обучение")
    for images, targets, target_lengths, texts in progress_bar:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # (batch, seq_len, num_classes)
        
        # Подготовка для CTC loss
        # outputs: (seq_len, batch, num_classes)
        outputs = outputs.permute(1, 0, 2)
        
        # Вычисляем длину последовательности на выходе
        input_lengths = torch.full(
            size=(outputs.size(1),),
            fill_value=outputs.size(0),
            dtype=torch.long
        ).to(device)
        
        # CTC loss
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Вычисляем точность
        with torch.no_grad():
            pred_outputs = model(images)
            decoded_preds = decode_predictions(pred_outputs, dataset)
            
            for pred, true in zip(decoded_preds, texts):
                if pred == true:
                    correct += 1
                total += 1
        
        # Обновляем прогресс-бар
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, dataset):
    """Валидация модели"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Валидация")
        for images, targets, target_lengths, texts in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            
            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)
            
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Декодируем предсказания
            pred_outputs = model(images)
            decoded_preds = decode_predictions(pred_outputs, dataset)
            
            for pred, true in zip(decoded_preds, texts):
                if pred == true:
                    correct += 1
                total += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def train_model(dataset_dir='dataset', epochs=50, batch_size=32, lr=0.001):
    """Основная функция обучения"""
    
    # Определяем символы (должны совпадать с main.py)
    russian_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    numbers = '0123456789'
    characters = russian_letters + numbers
    
    print("=" * 60)
    print("Обучение модели CRNN для распознавания капчи")
    print("=" * 60)
    print(f"Символы: {characters}")
    print(f"Количество символов: {len(characters)}")
    print(f"Эпох: {epochs}")
    print(f"Размер батча: {batch_size}")
    print(f"Learning rate: {lr}")
    print("=" * 60)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Загружаем данные
    print("\nЗагрузка датасета...")
    train_loader, val_loader, dataset = get_data_loaders(
        dataset_dir=dataset_dir,
        characters=characters,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=4
    )
    
    print(f"Обучающих примеров: {len(train_loader.dataset)}")
    print(f"Валидационных примеров: {len(val_loader.dataset)}")
    
    # Создаем модель
    # num_classes включает blank символ (индекс 0)
    model = CRNN(num_classes=len(characters) + 1).to(device)
    print(f"\nМодель создана. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Функция потерь (CTC Loss)
    # blank=0 соответствует blank символу
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Планировщик learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # История обучения
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Папка для сохранения моделей
    os.makedirs('models', exist_ok=True)
    
    best_val_acc = 0
    
    print("\nНачинаем обучение...\n")
    
    # Обучение
    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}")
        print("-" * 60)
        
        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, dataset
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device, dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Обновляем learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nРезультаты эпохи {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if old_lr != new_lr:
            print(f"  Learning rate уменьшен: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'characters': characters,
            }, 'models/best_model.pth')
            print(f"  ✓ Сохранена лучшая модель (Val Acc: {val_acc:.2f}%)")
        
        # Сохраняем чекпоинт каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'characters': characters,
            }, f'models/checkpoint_epoch_{epoch + 1}.pth')
    
    # Сохраняем финальную модель
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'characters': characters,
    }, 'models/final_model.pth')
    
    # Строим графики
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"Модель сохранена в: models/best_model.pth")
    print("=" * 60)


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Строит графики истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('История Loss')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('История Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\nГрафик истории обучения сохранен в: models/training_history.png")


if __name__ == "__main__":
    train_model(
        dataset_dir='dataset',
        epochs=50,
        batch_size=32,
        lr=0.001
    )

