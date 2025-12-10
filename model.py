import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) модель для распознавания капчи.
    Использует CNN для извлечения признаков и RNN (LSTM) для обработки последовательности.
    """
    
    def __init__(self, num_classes, img_height=80, img_width=200):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN для извлечения признаков
        # Выходной размер: (batch, 512, 1, 50) после всех слоев
        self.cnn = nn.Sequential(
            # Первый блок
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Второй блок
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Третий блок
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Четвертый блок
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Пятый блок
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Шестой блок
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Седьмой блок
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # RNN для обработки последовательности
        # После CNN размерность: (batch, 512, 1, feature_width)
        self.rnn_input_size = 512
        self.hidden_size = 256
        self.num_layers = 2
        
        self.rnn = nn.LSTM(
            self.rnn_input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Линейный слой для классификации
        # num_classes включает blank символ (индекс 0)
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN извлечение признаков
        conv_features = self.cnn(x)
        
        # Преобразуем размерность для RNN
        # (batch, channels, height, width) -> (batch, width, channels * height)
        batch_size, channels, height, width = conv_features.size()
        
        # Убираем dimension height - используем среднее по высоте или просто reshape
        # Если height > 1, усредняем по высоте или просто flatten
        if height > 1:
            # Вариант 1: Усредняем по высоте (Global Average Pooling по высоте)
            conv_features = conv_features.mean(dim=2)  # (batch, channels, width)
        else:
            # Если height == 1, просто убираем эту размерность
            conv_features = conv_features.squeeze(2)  # (batch, channels, width)
        
        # Транспонируем для RNN: (batch, width, channels)
        conv_features = conv_features.permute(0, 2, 1)
        
        # RNN
        rnn_out, _ = self.rnn(conv_features)
        
        # Классификация
        output = self.fc(rnn_out)  # (batch, width, num_classes)
        
        # Логиты для CTC loss
        output = output.log_softmax(2)
        
        return output


def decode_predictions(predictions, dataset):
    """
    Декодирует предсказания модели в текст.
    Использует CTC декодирование (убирает повторы и blank символы).
    """
    # predictions shape: (batch, seq_len, num_classes) - уже log_softmax
    # Получаем индексы наиболее вероятных символов
    pred_indices = predictions.argmax(dim=2)  # (batch, seq_len)
    
    decoded_texts = []
    for pred_seq in pred_indices:
        # Используем метод decode датасета для CTC декодирования
        text = dataset.decode(pred_seq)
        decoded_texts.append(text)
    
    return decoded_texts

