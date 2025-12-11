import os
import random
import numpy as np
from captcha.image import ImageCaptcha
from PIL import Image
from tqdm import tqdm


def apply_random_transparency(image):
    """
    Применяет случайную прозрачность к символам капчи.
    Иногда символы могут почти сливаться с фоном.

    Args:
        image: PIL Image (RGB)

    Returns:
        PIL Image (RGB) с примененной прозрачностью
    """
    # Конвертируем в numpy массив для работы с пикселями
    img_array = np.array(image)

    # Определяем фон (обычно белый или очень светлый)
    # Порог для определения фона (значения близкие к 255)
    background_threshold = 240

    # Создаем маску для символов (не фон)
    # Символы - это пиксели, где хотя бы один канал меньше порога
    is_symbol = np.any(img_array < background_threshold, axis=2)

    # Случайная прозрачность для символов (от 0.3 до 1.0)
    # Иногда делаем очень низкую прозрачность (0.3-0.5) для эффекта "слияния"
    transparency_prob = random.random()
    if transparency_prob < 0.2:  # 20% случаев - почти сливаются
        alpha_min, alpha_max = 0.3, 0.5
    elif transparency_prob < 0.5:  # 30% случаев - средняя прозрачность
        alpha_min, alpha_max = 0.5, 0.7
    else:  # 50% случаев - нормальная видимость
        alpha_min, alpha_max = 0.7, 1.0

    # Генерируем базовое значение прозрачности для всего изображения
    base_alpha = random.uniform(alpha_min, alpha_max)

    # Добавляем небольшие случайные вариации для более реалистичного эффекта
    # Используем небольшой шум, чтобы прозрачность была не совсем однородной
    noise = np.random.normal(0, 0.1, size=img_array.shape[:2])
    alpha_values = np.clip(base_alpha + noise, alpha_min, alpha_max)

    # Применяем прозрачность только к символам (фон остается непрозрачным)
    alpha_mask = np.where(is_symbol, alpha_values, 1.0)

    # Получаем цвет фона (средний цвет по краям изображения)
    # Берем пиксели по краям как фон
    edge_pixels = np.concatenate([
        img_array[0, :].reshape(-1, 3),
        img_array[-1, :].reshape(-1, 3),
        img_array[:, 0].reshape(-1, 3),
        img_array[:, -1].reshape(-1, 3)
    ], axis=0)
    background_color = np.mean(edge_pixels, axis=0)

    # Применяем прозрачность: смешиваем символы с фоном
    # Формула: result = symbol * alpha + background * (1 - alpha)
    alpha_3d = alpha_mask[:, :, np.newaxis]
    result_array = img_array * alpha_3d + background_color * (1 - alpha_3d)

    # Ограничиваем значения в диапазоне [0, 255]
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # Конвертируем обратно в PIL Image
    result_image = Image.fromarray(result_array, mode='RGB')

    return result_image


def generate_captcha_dataset(num_images=10):
    """
    Генерирует датасет из изображений капчи с русскими буквами и цифрами.
    Имя файла соответствует тексту на капче.

    Args:
        num_images: Количество изображений для генерации (по умолчанию 50000)
    """
    # Путь к папке для сохранения датасета
    dataset_dir = 'dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    # Русские буквы (заглавные) и цифры
    # Исключаем Ь, Ъ, Й, О, Щ - их нет в капче
    russian_letters = 'АБВГДЕЖИКЛМНПРСТУФХЦЧШЫЭЮЯ'
    numbers = '1234567890'
    characters = russian_letters + numbers

    # Длина текста на капче (можно варьировать)
    min_length = 4
    max_length = 6

    # Инициализация генератора капчи
    # Увеличиваем размеры для лучшей читаемости русских символов
    image_captcha = ImageCaptcha(
        width=200,
        height=80,
        # Пробуем использовать встроенные шрифты или системные
    )

    # Словарь для отслеживания уже созданных файлов (избегаем перезаписи)
    created_files = set()

    print(f"Начинаю генерацию {num_images} изображений капчи...")
    print(f"Символы: {characters}")
    print(f"Длина текста: {min_length}-{max_length} символов")
    print(f"Папка для сохранения: {dataset_dir}")

    # Генерация изображений с прогресс-баром
    for i in tqdm(range(num_images), desc="Генерация капчи"):
        # Выбираем случайную длину текста
        length = random.randint(min_length, max_length)

        # Генерируем случайный текст
        captcha_text = ''.join(random.choices(characters, k=length))

        # Формируем имя файла (если такой уже есть, добавляем номер)
        filename = f"{captcha_text}.png"
        filepath = os.path.join(dataset_dir, filename)

        # Если файл с таким именем уже существует, добавляем номер
        counter = 1
        while filename in created_files:
            filename = f"{captcha_text}_{counter}.png"
            filepath = os.path.join(dataset_dir, filename)
            counter += 1

        created_files.add(filename)

        # Генерируем изображение
        image = image_captcha.generate_image(captcha_text)

        # Применяем случайную прозрачность символов
        image = apply_random_transparency(image)

        # Сохранение изображения
        image.save(filepath)

    print(
        f"\n✓ Готово! Сгенерировано {num_images} изображений в папке '{dataset_dir}'")
    print(f"✓ Уникальных файлов: {len(created_files)}")


if __name__ == "__main__":
    generate_captcha_dataset()
