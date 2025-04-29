import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image
import os
import time
import webbrowser
import random
import torch
from diffusers import StableDiffusionPipeline
from tkinter import font as tkfont
import pygame
import sys
import warnings

# Инициализация pygame для звука
pygame.mixer.init()

class StableDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="auto"):
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.pipe = None
        
    def load_model(self):
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None
            ).to(self.device)
        
    def generate_image(self, prompt, save_path=None):
        if not self.pipe:
            self.load_model()
            
        with torch.inference_mode():
            result = self.pipe(prompt, num_inference_steps=25)
            image = result.images[0]
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
                
            return image

class App:
    def __init__(self, root):
        self.root = root
        
        # 1% шанс на пасхалку Windows 95
        if random.random() < 0.01:
            self.is_win95_mode = True
            self.root.title("You in 1995!")
            self.play_win95_sound()
        else:
            self.is_win95_mode = False
            self.root.title("AI Image Generator")
            
        self.root.geometry("900x700")
        self.setup_cursor()
        
        # Иконки
        self.icons = ["lamp.png", "lamp2.png", "link.png"]
        self.icon_images = []
        
        # Подсказки (разные для режимов)
        self.modern_tips = [
            "Did you know? Click 'Generate' to create your image!",
            "Pro Tip: More detailed descriptions yield better results!",
            "Try adding styles like: 'digital art', 'oil painting' or 'photorealistic'",
            "Hint: Separate different elements with commas",
            "For best results, describe both subject and background",
            "Tip: The AI understands adjectives like 'majestic', 'futuristic', 'vintage'",
            "Remember: You can specify lighting like 'dramatic lighting' or 'soft glow'",
            "Fun Fact: This uses cutting-edge neural network technology!*",
            "*Not actually available in 1995",
            "System Tip: Higher VRAM improves generation speed",
            "Windows 95 Tip: Describe what you want to see in detail",
            "Did you know? You can generate multiple variations",
            "Artist's Tip: Reference famous styles like 'Picasso' or 'Van Gogh'",
            "Expert Advice: Combine concepts like 'cyberpunk cityscape at night'",
            "Processing Tip: Complex images may take longer to render"
        ]
        
        self.win95_tips = [
            "Invested in 1995!",
            "System Tip: Close other programs to free up memory",
            "Windows 95 Tip: This is 32-bit protected mode!",
            "Microsoft® recommends 486DX/66MHz or higher"
        ]
        
        self.load_icons()
        self.setup_ui()
        self.model = None
        self.last_generation_time = 0

    def play_win95_sound(self):
        """Воспроизведение звука Windows 95"""
        try:
            sound = pygame.mixer.Sound(b'windows95_startup.wav')  # Замените на реальный звук
            sound.play()
        except:
            # Запасной вариант - системный бип
            print('\a')

    def setup_cursor(self):
        """Установка курсора (песочные часы для Win95)"""
        if self.is_win95_mode:
            self.root.config(cursor="watch")
        else:
            self.root.config(cursor="arrow")

    def load_icons(self):
        """Загрузка иконок с учетом режима"""
        self.icon_images = []
        icon_set = self.icons
        
        # В режиме Win95 можно использовать другие иконки
        if self.is_win95_mode:
            icon_set = ["mycomputer.png", "network.png", "recyclebin.png"]
            
        for icon in icon_set:
            try:
                img = Image.open(icon)
                img = img.resize((48, 48), Image.LANCZOS)
                self.icon_images.append(ImageTk.PhotoImage(img))
            except:
                img = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
                self.icon_images.append(ImageTk.PhotoImage(img))

    def setup_ui(self):
        """Настройка интерфейса в зависимости от режима"""
        if self.is_win95_mode:
            self.setup_win95_style()
        else:
            self.setup_modern_style()
            
        # Основной контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Панель ввода
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.prompt_entry = ttk.Entry(input_frame, width=60)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.generate_btn = ttk.Button(input_frame, text="Generate", command=self.generate)
        self.prompt_entry.bind('<Return>', lambda e: self.generate())
        self.generate_btn.pack(side=tk.RIGHT)
        
        # Область изображения
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.show_tip()
        
        # Панель статуса
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.time_label = ttk.Label(status_frame, text="Time: -")
        self.time_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        button_frame = ttk.Frame(status_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.open_folder_btn = ttk.Button(button_frame, text="Open Folder", command=self.open_temp_folder, state='disabled')
        self.open_folder_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_btn = ttk.Button(button_frame, text="Save", command=self.save_image, state='disabled')
        self.save_btn.pack(side=tk.RIGHT)
        
        # Кнопка возврата (только в режиме Win95)
        if self.is_win95_mode:
            bottom_frame = ttk.Frame(main_frame)
            bottom_frame.pack(fill=tk.X, pady=(10, 0))
            
            # Центрируем кнопку с отступами
            self.return_btn = ttk.Button(
                bottom_frame, 
                text="Return to 2025", 
                command=self.return_to_modern,
                width=15
            )
            self.return_btn.pack(pady=5, padx=10)  # Отступы сверху/снизу и по бокам
            
            # Стилизация для Win95
            self.return_btn.configure(style='TButton')

    def setup_modern_style(self):
        """Современный темный стиль"""
        self.root.configure(bg='#1e1e1e')
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('.', 
                      background='#1e1e1e',
                      foreground='#ffffff',
                      font=('Segoe UI', 10))
        
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TButton', background='#333333', foreground='#ffffff', borderwidth=0)
        style.map('TButton', background=[('active', '#444444')])
        style.configure('TEntry', fieldbackground='#2d2d2d', foreground='#ffffff')
        style.configure('Card.TFrame', background='#252526')

    def setup_win95_style(self):
        """Стиль Windows 95"""
        self.root.configure(bg='#008080')
        style = ttk.Style()
        style.theme_use('winnative')
        
        style.configure('.', 
                      background='#c0c0c0',
                      foreground='black',
                      font=('MS Sans Serif', 10))
        
        style.configure('TButton',
                      background='#c0c0c0',
                      relief='raised',
                      borderwidth=2)
        
        style.map('TButton',
                background=[('active', '#a0a0a0')])
        
        style.configure('Card.TFrame', background='#c0c0c0', relief='sunken', borderwidth=2)

    def show_tip(self):
        """Показ подсказки с иконками"""
        self.canvas.delete("all")
        
        # Выбор подсказки в зависимости от режима
        tips = self.win95_tips if self.is_win95_mode else self.modern_tips
        tip = random.choice(tips)
        
        # Настройки стиля
        if self.is_win95_mode:
            bg_color = '#c0c0c0'
            text_color = 'black'
            outline_color = '#808080'
            font = ('MS Sans Serif', 12)
        else:
            bg_color = '#252526'
            text_color = '#aaaaaa'
            outline_color = '#3e3e42'
            font = ('Segoe UI', 12)
        
        # Отрисовка
        self.canvas.configure(bg=bg_color)
        self.canvas.create_rectangle(50, 50, 850, 450, 
                                   fill=bg_color, 
                                   outline=outline_color, 
                                   width=2)
        
        self.canvas.create_text(450, 250, 
                              text=tip, 
                              font=font, 
                              width=700, 
                              fill=text_color, 
                              justify='center')
        
        # Иконки
        if self.icon_images:
            positions = [(150, 150), (750, 150), (150, 350), (750, 350)]
            for pos in random.sample(positions, 2):
                icon = random.choice(self.icon_images)
                self.canvas.create_image(pos[0], pos[1], image=icon, anchor='center')

    def return_to_modern(self):
        """Эффектный возврат в современность"""
        # Анимация исчезновения
        for i in range(5, 0, -1):
            self.return_btn.configure(text=f"Returning in {i}...")
            self.root.update()
            time.sleep(0.3)
        
        # Взрывной эффект (псевдо-анимация)
        self.canvas.delete("all")
        self.canvas.create_text(450, 250, 
                            text="*woosh*", 
                            font=("Comic Sans MS", 48), 
                            fill="cyan")
        self.root.update()
        time.sleep(0.5)
        
        # Полный сброс стиля
        self.is_win95_mode = False
        self.setup_modern_style()
        self.show_tip()
        self.root.title("AI Image Generator")
        self.root.config(cursor="arrow")
        self.return_btn.destroy()
        
        # Современный звук (опционально)
        try:
            pygame.mixer.Sound(b'modern_sound.wav').play()
        except:
            print('\a')  # Системный бип
    def generate(self, event=None):
        """Генерация изображения по запросу"""
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showwarning("Error", "Please enter a prompt")
            return
            
        self.status_label.config(text="Initializing model...")
        self.generate_btn.config(state='disabled')
        self.root.update()
        
        try:
            start_time = time.time()
            
            # Ленивая загрузка модели
            if not self.model:
                self.model = StableDiffusion()
                
            self.status_label.config(text="Generating image...")
            
            # Создаем временную папку
            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", "generated.png")
            
            # Генерация изображения
            img = self.model.generate_image(prompt, save_path=temp_path)
            
            # Отображение результата
            self.display_image(img)
            
            # Обновление интерфейса
            self.last_generation_time = time.time() - start_time
            self.time_label.config(text=f"Time: {self.last_generation_time:.1f}s")
            self.status_label.config(text="Done!")
            self.save_btn.config(state='normal')
            self.open_folder_btn.config(state='normal')
            
        except Exception as e:
            self.status_label.config(text="Error")
            messagebox.showerror("Error", str(e))
            if hasattr(self, 'show_tip'):
                self.show_tip()  # Показываем подсказку при ошибке
        finally:
            self.generate_btn.config(state='normal')            
    def display_image(self, img):
        """Отображение изображения с масштабированием"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()    
        # Пропорциональное масштабирование
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
            
        if canvas_ratio > img_ratio:
            new_height = canvas_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / img_ratio)
                
        img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Обновление холста
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
        canvas_width//2,
        canvas_height//2,
        image=self.tk_img
        )
    
    def save_image(self):
        """Сохранение изображения в выбранное место"""
        temp_path = os.path.join("temp", "generated.png")
        if not os.path.exists(temp_path):
            messagebox.showerror("Error", "No image to save")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("All files", "*.*")
            ],
            title="Save Image"
        )
        
        if save_path:
            try:
                img = Image.open(temp_path)
                img.save(save_path)
                messagebox.showinfo("Success", f"Image saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{str(e)}")
    
    def open_temp_folder(self):
        """Открытие папки с временными файлами в проводнике"""
        temp_path = os.path.abspath("temp")
        if not os.path.exists(temp_path):
            messagebox.showerror("Error", "Folder not found")
            return
            
        try:
            webbrowser.open(f"file://{temp_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open folder:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()