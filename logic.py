from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

class StableDiffusion:
    """Класс для генерации изображений с помощью Stable Diffusion"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        """
        Инициализация пайплайна
        :param model_id: ID модели из Hugging Face Hub
        :param device: устройство для вычислений ("cuda" или "cpu")
        """
        self.device = device
        self.model_id = model_id
        
        # Определяем тип данных в зависимости от устройства
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"⏳ Загрузка модели {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype
        ).to(device)
        print("✅ Модель загружена и готова к использованию!")
    
    def generate_image(self, prompt, save_path=None):
        """
        Генерация изображения по текстовому описанию
        :param prompt: текстовый запрос
        :param save_path: путь для сохранения (None - не сохранять)
        :return: PIL.Image объект
        """
        print(f"🎨 Генерация изображения по запросу: '{prompt}'")
        
        with torch.inference_mode():
            result = self.pipe(prompt)
            image = result.images[0]
        
        if save_path:
            image.save(save_path)
            print(f"💾 Изображение сохранено как {save_path}")
        
        return image