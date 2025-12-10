import os
import random
from captcha.image import ImageCaptcha
from tqdm import tqdm


def generate_captcha_dataset():
    """
    Генерирует датасет из 50000 изображений капчи с русскими буквами и цифрами.
    Имя файла соответствует тексту на капче.
    """
    # Путь к папке для сохранения датасета
    dataset_dir = 'dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    # Русские буквы (заглавные) и цифры
    russian_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    numbers = '0123456789'
    characters = russian_letters + numbers

    # Длина текста на капче (можно варьировать)
    min_length = 4
    max_length = 6

    # Количество изображений
    num_images = 50000

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

        # Сохранение изображения
        image.save(filepath)

    print(
        f"\n✓ Готово! Сгенерировано {num_images} изображений в папке '{dataset_dir}'")
    print(f"✓ Уникальных файлов: {len(created_files)}")


if __name__ == "__main__":
    generate_captcha_dataset()
