import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CRNN, decode_predictions
from dataset_loader import CaptchaDataset
import argparse
import os


def load_model(model_path, characters, device):
    """Загружает обученную модель"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # num_classes включает blank символ
    model = CRNN(num_classes=len(characters) + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('characters', characters)


def predict_image(image_path, model, dataset, device):
    """Предсказывает текст на изображении капчи"""
    # Загружаем и обрабатываем изображение
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        decoded_text = decode_predictions(outputs, dataset)
    
    return decoded_text[0]


def main():
    parser = argparse.ArgumentParser(description='Распознавание капчи')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению капчи')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Путь к модели')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Путь к датасету (для получения словаря символов)')
    
    args = parser.parse_args()
    
    # Определяем символы
    russian_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    numbers = '0123456789'
    characters = russian_letters + numbers
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    print(f"Загрузка модели из {args.model}...")
    model, model_characters = load_model(args.model, characters, device)
    print("Модель загружена!")
    
    # Создаем dataset для декодирования (используем символы из модели если они есть)
    if model_characters:
        characters = model_characters
    
    dataset = CaptchaDataset(args.dataset_dir, characters, transform=None)
    
    # Предсказание
    print(f"\nРаспознавание изображения: {args.image}")
    predicted_text = predict_image(args.image, model, dataset, device)
    
    # Если имя файла содержит правильный ответ, показываем его
    filename = os.path.basename(args.image)
    if filename.endswith('.png'):
        correct_text = filename.replace('.png', '').rsplit('_', 1)[0]
        print(f"Правильный ответ: {correct_text}")
        print(f"Предсказание:     {predicted_text}")
        
        if predicted_text == correct_text:
            print("✓ Правильно!")
        else:
            print("✗ Неправильно")
    else:
        print(f"Предсказание: {predicted_text}")


if __name__ == "__main__":
    main()

