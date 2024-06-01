import os                       # Для работы с файловой системой
import csv                      # Для работы с CSV файлами
import cv2                      # OpenCV для работы с видео и изображениями
from ultralytics import YOLO    # Библиотека Ultralytics для работы с YOLO моделями
import subprocess               # Для запуска внешних команд, например, ffmpeg
import datetime


def processing(
        video_file_path: str, reports_folder_path: str,
        model: YOLO, confidence: float,
        buffer_time: int, skip_frames: int,
        window_tk, text_widget, pb_widget
    ):

    dir_path, filename = os.path.split(video_file_path)
    os.makedirs(os.path.join(reports_folder_path, filename), exist_ok=True)

    # Создаём пустой CSV файлы в начале работы, чтобы избежать ошибок при его отсутствии
    csv_file = os.path.join(reports_folder_path, f"{filename}_timestamps.csv")
    open(csv_file, 'w').close()

    # Основная функция для детекции фрагментов видео
    detect_video_fragments(video_path=video_file_path, model=model, confidence=confidence,
                           buffer_time=buffer_time, skip_frames=skip_frames, output_csv=csv_file,
                           window_tk=window_tk, text_widget=text_widget, pb_widget=pb_widget)

    # Нарезаем фрагменты видео по меткам времени
    slice_video_fragments(reports_folder_path, csv_file, video_file_path)


# Основная функция для детекции фрагментов видео
def detect_video_fragments(video_path: str, model: YOLO, confidence: float,
                           buffer_time: int, skip_frames: int, output_csv: str,
                           window_tk, text_widget, pb_widget):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pb_widget.config(value=0, maximum=total_frames)
    detection_active = False  # Флаг детекции
    start_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_frames == 0:
            result = model.predict(frame, save=False, conf=confidence, classes=[0], verbose=False)

            # Проверка и обновление детекций
            if any(len(res.boxes) > 0 for res in result):
                if not detection_active:
                    start_frame = frame_index
                    detection_active = True  # Началась регистрация нарушения

                last_detection_frame_index = frame_index

            # Проверка условий завершения фрагмента
            if detection_active and (frame_index - last_detection_frame_index) > buffer_time * fps:
                save_detected_fragment(start_frame, last_detection_frame_index, buffer_time, fps, output_csv,
                                       window_tk, text_widget)
                detection_active = False  # Закончилась регистрация нарушения

        frame_index += 1

        # Обновляем прогресс-бар
        pb_widget.config(value=frame_index)
        window_tk.update()

    # Сохранение последнего фрагмента
    if detection_active:
        save_detected_fragment(start_frame, frame_index, buffer_time, fps, output_csv, window_tk, text_widget)

    cap.release()


def create_folder_with_timestamp(local_base_path):
    # Получаем текущую дату и время
    current_datetime = datetime.datetime.now()

    # Форматируем дату и время в нужный формат
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")

    # Создаем имя папки с префиксом и временной меткой
    starting_folder_name = f"starting_in_{timestamp}"

    # Создаем полный путь к папке
    starting_folder_path = os.path.join(local_base_path, starting_folder_name)

    # Создаем папку с отчётом
    os.makedirs(starting_folder_path, exist_ok=True)

    return starting_folder_path


# Обновляем текстовый виджет (добавляем текст в конец)
def update_text_widget(window, text_widget, update_text):
    # Включение редактирования
    text_widget.config(state='normal')

    # Добавление текста о количестве найденных видео-файлов
    text_widget.insert('end', update_text + '\n')

    # Отключение редактирования
    text_widget.config(state='disabled')

    # Прокрутка вниз
    text_widget.yview_moveto(1.0)  # Устанавливаем Scrollbar внизу

    # Принудительное обновление интерфейса
    window.update()
    # window.update_idletasks()


# Переводим номера кадров во временные метки
def interval_to_time(start, stop, fps):

    # Внутренняя функция для перевода секунд в формат часы:минуты:секунды
    def _seconds_to_time(s):
        hours = int(s // 3600)
        minutes = int(s % 3600 // 60)
        seconds = s % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"

    start_time = _seconds_to_time(start / fps)
    end_time = _seconds_to_time(stop / fps)
    return start_time, end_time


# Сохраняет обнаруженные интервалы в CSV файл
def save_detected_fragment(start_frame, end_frame, buffer_time, fps, output_csv, window_tk, text_widget):
    if (end_frame - start_frame) >= buffer_time * fps:
        start_time, end_time = interval_to_time(start_frame, end_frame, fps)
        with open(output_csv, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([start_time, end_time])

        update_text = f"- нарушение в промежутке времени: {start_time} - {end_time}"
        update_text_widget(window_tk, text_widget, update_text)


# Нарезаем фрагменты видео по меткам времени
def slice_video_fragments(output_folder, csv_file, video_path):
    # Читаем временные метки из CSV файла и нарезает видео
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for start_time, end_time in reader:
            start_time_formatted = start_time.replace(":", "-")
            end_time_formatted = end_time.replace(":", "-")
            output_path = os.path.join(output_folder, f"fragment_{start_time_formatted}_{end_time_formatted}.mp4")

            # Используем ffmpeg для вырезания фрагментов видео
            command = ['ffmpeg', '-i', video_path, '-ss', start_time, '-to', end_time, '-c', 'copy', output_path, '-y']
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def export_text_to_file(text_widget, reports_folder_path):
    # Получаем имя последней папки в пути starting_folder_path
    folder_name = os.path.basename(reports_folder_path)

    # Формируем путь к файлу txt
    output_file_path = os.path.join(reports_folder_path, f"log_{folder_name.replace('запуск', 'обработки')}.txt")

    # Получаем текст из виджета
    text = text_widget.get("1.0", "end-1c")  # Получаем текст с первой строки до последней без символа перевода строки

    # Записываем текст в файл
    with open(output_file_path, "w") as file:
        file.write(text)

    return output_file_path
