import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image
import os
import platform
from logic import StableDiffusion

class StableDiffusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stable Diffusion Image Generator")
        self.root.geometry("800x600")
        
        # Инициализация модели (по умолчанию на CPU, можно изменить на "cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusion(device=self.device)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Фрейм для ввода запроса
        input_frame = ttk.LabelFrame(self.root, text="Введите запрос для генерации изображения", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.prompt_entry = ttk.Entry(input_frame, width=70)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.prompt_entry.bind("<Return>", lambda e: self.generate_image())
        
        generate_btn = ttk.Button(input_frame, text="Сгенерировать", command=self.generate_image)
        generate_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Фрейм для отображения изображения
        self.image_frame = ttk.LabelFrame(self.root, text="Результат", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        save_btn = ttk.Button(button_frame, text="Сохранить изображение", command=self.save_image)
        save_btn.pack(side=tk.LEFT)
        
        open_folder_btn = ttk.Button(button_frame, text="Открыть папку с сохранениями", command=self.open_save_folder)
        open_folder_btn.pack(side=tk.RIGHT)
        
        # Переменные для хранения изображений
        self.current_image = None
        self.current_image_path = None
        
    def generate_image(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showwarning("Ошибка", "Пожалуйста, введите текстовый запрос")
            return
        
        try:
            # Генерация изображения
            image = self.model.generate_image(prompt)
            
            # Сохранение временного файла для отображения
            if not os.path.exists("temp"):
                os.makedirs("temp")
            temp_path = os.path.join("temp", "temp_output.png")
            image.save(temp_path)
            self.current_image_path = temp_path
            
            # Отображение изображения
            self.display_image(temp_path)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при генерации изображения:\n{str(e)}")
    
    def display_image(self, image_path):
        img = Image.open(image_path)
        
        # Получаем размеры фрейма для отображения
        frame_width = self.image_frame.winfo_width() - 20
        frame_height = self.image_frame.winfo_height() - 20
        
        # Масштабируем изображение под размер фрейма
        img.thumbnail((frame_width, frame_height), Image.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Сохраняем ссылку, чтобы избежать сборки мусора
        
    def save_image(self):
        if not self.current_image_path:
            messagebox.showwarning("Ошибка", "Сначала сгенерируйте изображение")
            return
            
        # Предлагаем пользователю выбрать место сохранения
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Сохранить изображение как"
        )
        
        if save_path:
            try:
                img = Image.open(self.current_image_path)
                img.save(save_path)
                self.current_image_path = save_path
                messagebox.showinfo("Успех", f"Изображение успешно сохранено в:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{str(e)}")
    
    def open_save_folder(self):
        if not self.current_image_path:
            messagebox.showwarning("Ошибка", "Нет сохраненного изображения")
            return
            
        folder_path = os.path.dirname(self.current_image_path)
        
        try:
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":
                os.system(f"open '{folder_path}'")
            else:  # Linux
                os.system(f"xdg-open '{folder_path}'")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку:\n{str(e)}")

if __name__ == "__main__":
    import torch  # Импортируем torch здесь, чтобы проверить доступность CUDA
    
    root = tk.Tk()
    app = StableDiffusionApp(root)
    root.mainloop()