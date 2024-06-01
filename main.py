import shutil

import torch
from tkinter import *
from tkinter import ttk, filedialog
import os
from ultralytics import YOLO
#from detection import *


def open_dir(dir_path):
    if dir_path:
        os.startfile(dir_path)


class Application:
    def __init__(self):
        self.version = "0.1"
        self.color_rzd = "#E21A1A"  # Фирменный цвет РЖД
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Создаём главное окно
        self.root = Tk()

        # Получаем путь к файлу main.py
        self.main_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.main_path, 'data')

        # Переменные хранящие пути к папкам
        self.dir_with_files_for_processing = StringVar(value=os.path.join(self.data_path, "files_for_processing"))
        self.dir_with_processed_files = StringVar(value=os.path.join(self.data_path, "processed_files"))
        self.dir_with_weight = StringVar(value=os.path.join(self.data_path, "weights"))
        self.dir_with_reports = os.path.join(self.data_path, "reports")

        # Путь к весам YOLO
        self.weights_path = os.path.join(self.dir_with_weight.get(), "weights.pt")
        self.confidence = 0.35  # Порог обнаружения
        self.skip_frames = 5  # Количество пропускаемых кадров при первичной детекции
        self.buffer_time = 3  # Количество секунд ожидания следующей детекции

        self.list_of_video_files_for_processing = []  # Список файлов для обработки
        self.numbers_of_video_files_for_processing = IntVar(value=0)  # Количество файлов для обработки

        # Сканируем папку с файлами для обработки (dir_with_files_for_processing)
        self._scan_dir_with_files_for_processing()

        # Переменные для прогресс-баров
        self.progressbar_task_value = IntVar(value=0)
        self.progressbar_total_value = IntVar(value=0)

        # Задаём строки для элементов
        self.row_logo = 0
        self.row_left_right_frames = 1
        self.row_progressbars = 2

        # Заголовок и иконка главного окна
        self.root.title(f"Автоматизация выявления технологических нарушений v.{self.version}")
        icon_path = os.path.join(self.data_path, "ico2.png")
        self.root.iconphoto(True, PhotoImage(file=icon_path))

        # Отрисовываем логотип РЖД в верхней части окна
        logo_img = PhotoImage(file=os.path.join(self.data_path, "logo_rzd_2.png"))
        label = Label(self.root, image=logo_img, background=self.color_rzd)
        label.grid(row=self.row_logo, column=0, columnspan=2, sticky=NW)

        # Наполняем окно виджетами
        self._configure_main_window()
        self._create_menu()
        self._create_left_frame()
        self._create_right_frame()
        self._create_progressbars()

        # Запуск главного цикла обработки событий Tkinter
        self.root.mainloop()

    def _configure_main_window(self):
        self.common_pad = 10  # Общий отступ, где это применяется
        self.root.config(bg=self.color_rzd, padx=self.common_pad / 2, pady=self.common_pad / 2)

        # Задаём размеры и позиционирование окна
        self.window_width = 1125
        self.window_height = 690
        pos_x = int(self.root.winfo_screenwidth() / 2 - self.window_width / 2)
        pos_y = int(self.root.winfo_screenheight() / 2 - self.window_height / 2)
        self.root.geometry(f"{self.window_width}x{self.window_height}+{pos_x}+{pos_y}")
        self.root.resizable(False, False)

        # Создание объекта Style
        style = ttk.Style(self.root)

        # Установка темы ('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
        #style.theme_use("xpnative")  # Используем тему "xpnative"

    def _create_menu(self):
        # Создание главного меню
        main_menu = Menu(self.root)
        self.root.config(menu=main_menu)

        # Создание меню "Файл"
        file_menu = Menu(main_menu, tearoff=0)
        file_menu.add_command(label="Папка с файлами для обработки")
        file_menu.add_command(label="Выбрать веса модели")
        file_menu.add_separator()  # Черта-разделитель
        file_menu.add_command(label="Сохранить параметры")
        file_menu.add_separator()  # Черта-разделитель
        file_menu.add_command(label="Выход")

        main_menu.add_cascade(label="Файл", menu=file_menu)

        # Создание меню "Справка"
        help_menu = Menu(main_menu, tearoff=0)
        help_menu.add_command(label="Справка")  # Добавление команды для открытия документа .docx
        help_menu.add_separator()  # Черта-разделитель
        help_menu.add_command(label="О приложении")  # Создание пункта "О приложении"
        main_menu.add_cascade(label="Справка", menu=help_menu)

    def _create_left_frame(self):
        # Создаём левый фрейм для папки с файлами
        left_frame = Frame(self.root, padx=self.common_pad, pady=self.common_pad,
                           highlightbackground=self.color_rzd, highlightthickness=5)
        left_frame.grid(row=self.row_left_right_frames, column=0, sticky=NSEW)

        # Создаём текстовый фрейм для путей к папкам
        path_label_frame = LabelFrame(left_frame, text="Пути к папкам")
        path_label_frame.grid(row=0, column=0, sticky=NSEW)

        # Наполняем текстовый фрейм для путей к папкам
        label = Label(path_label_frame, text="Путь к папке с видеофайлами для обработки:")
        label.grid(row=0, column=0, columnspan=2, padx=self.common_pad, pady=self.common_pad, sticky=W)

        button = Button(path_label_frame, text="Открыть",
                        command=lambda: open_dir(self.dir_with_files_for_processing.get()))
        button.grid(row=1, column=0, padx=self.common_pad, sticky=W)

        label = Label(path_label_frame, textvariable=self.dir_with_files_for_processing,
                      justify=LEFT, wraplength=self.window_width / 4)
        label.grid(row=1, column=1, padx=self.common_pad, sticky=NW)

        label = Label(path_label_frame)
        label.grid(row=2, column=0, columnspan=2, sticky=W)

        label = Label(path_label_frame, text="Путь к папке с обработанными видеофайлами:")
        label.grid(row=3, column=0, columnspan=2, padx=self.common_pad, pady=self.common_pad, sticky=W)

        button = Button(path_label_frame, text="Открыть",
                        command=lambda: open_dir(self.dir_with_processed_files.get()))
        button.grid(row=4, column=0, padx=self.common_pad, sticky=W)

        label = Label(path_label_frame, textvariable=self.dir_with_processed_files,
                      justify=LEFT, wraplength=self.window_width / 4)
        label.grid(row=4, column=1, padx=self.common_pad, sticky=W)

        label = Label(path_label_frame)
        label.grid()

        label = Label(left_frame)
        label.grid(row=1)

        # Создаём текстовый фрейм для запуска обработки
        run_label_frame = LabelFrame(left_frame, text="Обработка видеофайлов", padx=self.common_pad)
        run_label_frame.grid(row=2, column=0, sticky=NSEW)

        label = Label(run_label_frame, text="Количество видеофайлов в папке для обработки:", justify=LEFT)
        label.grid(row=0, column=0, pady=self.common_pad, sticky=W)

        label = Label(run_label_frame, textvariable=self.numbers_of_video_files_for_processing, justify=LEFT)
        label.grid(row=0, column=1, pady=self.common_pad, sticky=W)

        button = Button(run_label_frame, text="Выполнить обработку", font="bold", command=self._start_processing)
        button.grid(columnspan=2, pady=self.common_pad, sticky=NSEW)

        label = Label(run_label_frame)
        label.grid()

    def _create_right_frame(self):
        # Создаём правый фрейм для журнала
        right_frame = Frame(self.root, padx=self.common_pad, pady=self.common_pad,
                            highlightbackground=self.color_rzd, highlightthickness=5)
        right_frame.grid(row=self.row_left_right_frames, column=1, sticky=NSEW)

        # Создаём текстовый фрейм для журнала
        log_label_frame = LabelFrame(right_frame, text="Журнал обработки видео-файлов")
        log_label_frame.grid(row=0, column=0, sticky=NSEW)

        # Создание виджета Text
        self.text_widget = Text(log_label_frame, wrap='word', state='disabled', height=18)
        self.text_widget.grid(row=0, column=0)
        self.text_widget.insert(1.0, "First text")

        scroll = Scrollbar(log_label_frame, command=self.text_widget.yview)
        scroll.grid(row=0, column=1, sticky=NS)
        self.text_widget.config(yscrollcommand=scroll.set)

        button = Button(right_frame, text="Открыть папку с отчётами",
                        command=lambda: open_dir(self.dir_with_reports))
        button.grid(pady=self.common_pad, sticky=SW)

    def _create_progressbars(self):
        down_frame = Frame(self.root, padx=self.common_pad, pady=self.common_pad,
                           highlightbackground=self.color_rzd, highlightthickness=5)
        down_frame.grid(row=self.row_progressbars, column=0, columnspan=2, sticky=EW)

        progress_label_frame = LabelFrame(down_frame, text="Прогресс выполнения обработки")
        progress_label_frame.grid(sticky=EW)

        label = Label(progress_label_frame, text="Прогресс выполнения одного файла:", justify=LEFT)
        label.grid(row=0, column=0, padx=self.common_pad, pady=self.common_pad, sticky=W)

        self.progressbar_task = ttk.Progressbar(progress_label_frame, length=1060)
        self.progressbar_task.grid(row=1, column=0, columnspan=2, padx=self.common_pad, sticky=EW)

        label = Label(progress_label_frame, text="Прогресс выполнения всех файлов:", justify=LEFT)
        label.grid(row=3, column=0, padx=self.common_pad, pady=self.common_pad, sticky=W)

        self.progressbar_total = ttk.Progressbar(progress_label_frame, length=1060,
                                                 variable=self.progressbar_total_value)
        self.progressbar_total.grid(row=4, column=0, columnspan=2, padx=self.common_pad, sticky=EW)

        label = Label(progress_label_frame, text=f"Обработка производится на {self.device}")
        label.grid(row=5, column=0, padx=self.common_pad, pady=self.common_pad, sticky=W)

    def _scan_dir_with_files_for_processing(self):
        dir_with_files_for_processing = self.dir_with_files_for_processing.get()
        list_dir = os.listdir(dir_with_files_for_processing)
        list_of_video_files = []

        for file_name in list_dir:
            if file_name.lower().endswith(('.avi', '.mp4', '.mkv', '.mov', '.mpeg')):
                video_file_path = os.path.join(dir_with_files_for_processing, file_name)
                list_of_video_files.append(video_file_path)

        self.list_of_video_files_for_processing = list_of_video_files
        self.numbers_of_video_files_for_processing.set(len(list_of_video_files))

    def _start_processing(self):
        self._scan_dir_with_files_for_processing()

        start_time = datetime.datetime.now().replace(microsecond=0)
        start_text = f"Дата и время начала обработки: {start_time}"
        update_text_widget(self.root, self.text_widget, start_text)
        reports_folder_path = create_folder_with_timestamp(self.dir_with_reports)

        # Очистка содержимого текста в виджет Text()
        self.text_widget.delete('1.0', 'end')

        # Вывод текста в виджет Text()
        start_text = f"\nНайдено видеофайлов для обработки: {self.numbers_of_video_files_for_processing.get()}"
        update_text_widget(self.root, self.text_widget, start_text)
        self.progressbar_total.config(maximum=self.numbers_of_video_files_for_processing.get())

        model = YOLO(self.weights_path).to(self.device)

        for video_file_path in self.list_of_video_files_for_processing:
            _, file_name = os.path.split(video_file_path)

            update_text = f"\nСтарт обработки файла: {file_name}"
            update_text_widget(self.root, self.text_widget, update_text)

            # Получаем имя файла без расширения
            file_name_wo_ext = os.path.splitext(file_name)[0]
            reports_folder_path_for_file = os.path.join(reports_folder_path, file_name_wo_ext)
            processing(video_file_path=video_file_path, reports_folder_path=reports_folder_path_for_file,
                       model=model, confidence=self.confidence,
                       skip_frames=self.skip_frames, buffer_time=self.buffer_time,
                       window_tk=self.root, text_widget=self.text_widget, pb_widget=self.progressbar_task)

            self.progressbar_total_value.set(self.progressbar_total_value.get() + 1)

            # Перемещаем обработанный файл
            shutil.move(video_file_path, os.path.join(self.dir_with_processed_files.get(), file_name))
            self._scan_dir_with_files_for_processing()

        end_time = datetime.datetime.now().replace(microsecond=0)
        end_text = f"\nДата и время окончания обработки: {start_time}"
        update_text_widget(self.root, self.text_widget, end_text)

        end_text = f"\nВсего потрачено времени: {end_time - start_time}"
        update_text_widget(self.root, self.text_widget, end_text)

        export_text_to_file(text_widget=self.text_widget, reports_folder_path=reports_folder_path)


if __name__ == '__main__':
    app = Application()  # Создание экземпляра приложения
