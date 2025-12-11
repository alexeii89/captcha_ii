import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CaptchaDataset(Dataset):
    """
    Датасет для загрузки изображений капчи.
    Имя файла содержит ответ (текст на капче).
    """

    def __init__(self, dataset_dir, characters, transform=None):
        self.dataset_dir = dataset_dir
        self.characters = characters
        self.transform = transform

        # Создаем словарь для преобразования символов в индексы
        # Индексы начинаются с 1, так как 0 зарезервирован для blank в CTC loss
        self.char_to_idx = {char: idx + 1 for idx,
                            char in enumerate(characters)}
        self.idx_to_char = {idx: char for char,
                            idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank символ для CTC
        self.num_classes = len(characters) + 1  # +1 для blank символа

        # Загружаем список всех файлов
        self.image_files = [f for f in os.listdir(
            dataset_dir) if f.endswith('.png')]

        print(f"Загружено {len(self.image_files)} изображений")
        print(f"Количество символов: {len(characters)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        filepath = os.path.join(self.dataset_dir, filename)

        # Загружаем изображение
        image = Image.open(filepath).convert('RGB')

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)

        # Извлекаем текст из имени файла (убираем .png и возможные суффиксы _1, _2 и т.д.)
        text = filename.replace('.png', '').rsplit('_', 1)[0]

        # Преобразуем текст в тензор индексов
        # Пропускаем символы, которых нет в словаре (с предупреждением)
        target = []
        for char in text:
            if char in self.char_to_idx:
                target.append(self.char_to_idx[char])
            else:
                # Пропускаем неизвестный символ с предупреждением (только один раз)
                if not hasattr(self, '_warned_chars'):
                    self._warned_chars = set()
                if char not in self._warned_chars:
                    print(
                        f"Предупреждение: символ '{char}' не найден в словаре символов. Пропускаем.")
                    self._warned_chars.add(char)

        if len(target) == 0:
            # Если все символы были пропущены, создаем пустой target с одним blank символом
            target = [0]

        target_length = len(target)

        return image, torch.tensor(target, dtype=torch.long), target_length, text

    def decode(self, indices):
        """Преобразует индексы обратно в текст (с CTC декодированием - убирает blank и повторы)"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        # Убираем blank (0) и повторы
        decoded = []
        prev_idx = None
        for idx in indices:
            if idx != 0 and idx != prev_idx and idx < len(self.char_to_idx) + 1:
                char = self.idx_to_char.get(idx, '')
                if char:
                    decoded.append(char)
            prev_idx = idx
        return ''.join(decoded)


class DatasetWrapper(torch.utils.data.Dataset):
    """Обёртка для подмножества датасета с доступом к методам оригинального датасета"""

    def __init__(self, subset, original_dataset):
        self.subset = subset
        self.dataset = original_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

    # Проксируем важные методы и атрибуты
    @property
    def char_to_idx(self):
        return self.dataset.char_to_idx

    @property
    def idx_to_char(self):
        return self.dataset.idx_to_char

    @property
    def num_classes(self):
        return self.dataset.num_classes

    def decode(self, indices):
        return self.dataset.decode(indices)


def get_data_loaders(dataset_dir, characters, batch_size=32, train_split=0.8, num_workers=None):
    """
    Создает DataLoader для обучения и валидации.

    Args:
        dataset_dir: Путь к папке с датасетом
        characters: Строка со всеми возможными символами
        batch_size: Размер батча
        train_split: Доля данных для обучения (остальное для валидации)
        num_workers: Количество процессов для загрузки данных (None = автоматический выбор)
    """
    import platform

    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Создаем полный датасет
    full_dataset = CaptchaDataset(dataset_dir, characters, transform=transform)

    # Разделяем на train и val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Создаем обёртки для доступа к методам оригинального датасета
    train_dataset = DatasetWrapper(train_subset, full_dataset)
    val_dataset = DatasetWrapper(val_subset, full_dataset)

    # Автоматически определяем оптимальное количество workers
    if num_workers is None:
        # На macOS часто лучше использовать 0 для избежания проблем с multiprocessing
        if platform.system() == 'Darwin':
            num_workers = 0
        else:
            num_workers = 4

    # Создаем DataLoader
    # На macOS с MPS pin_memory не поддерживается, поэтому устанавливаем False
    use_pin_memory = torch.cuda.is_available() and not (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader, full_dataset


def collate_fn(batch):
    """
    Функция для объединения батча с переменной длиной последовательностей.
    """
    images, targets, target_lengths, texts = zip(*batch)

    # Стек изображений
    images = torch.stack(images, 0)

    # Объединяем все целевые последовательности в один тензор
    # Находим максимальную длину в батче
    max_len = max(target_lengths)

    # Создаем тензор с заполнением
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, (target, length) in enumerate(zip(targets, target_lengths)):
        padded_targets[i, :length] = target

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, padded_targets, target_lengths, texts
