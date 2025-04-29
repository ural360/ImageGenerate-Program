import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import warnings

class StableDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="auto"):
        """Инициализация с автоматическим выбором устройства"""
        self.device = self._get_device(device)
        self.model_id = model_id
        self.model_path = f"models/{model_id.split('/')[-1]}"
        
        # Проверяем и загружаем модель
        if not self._model_exists():
            self._download_model()
            
        self.pipe = self._load_model()

    def _get_device(self, device):
        """Автовыбор устройства (CUDA/CPU)"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _model_exists(self):
        """Проверяем наличие локальной модели"""
        required_files = ["model_index.json", "unet/diffusion_pytorch_model.bin"]
        return all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files)

    def _download_model(self):
        """Скачивание модели с прогрессом"""
        from huggingface_hub import snapshot_download
        os.makedirs(self.model_path, exist_ok=True)
        
        print("⏳ Скачивание модели (это займет несколько минут и ~7ГБ места)...")
        snapshot_download(
            self.model_id,
            local_dir=self.model_path,
            ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
            resume_download=True
        )
        print("✅ Модель загружена!")

    def _load_model(self):
        """Загрузка модели с предупреждениями"""
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                local_files_only=True
            ).to(self.device)

    def generate_image(self, prompt, save_path=None):
        """Генерация изображения"""
        with torch.inference_mode():
            result = self.pipe(prompt, num_inference_steps=25)
            img = result.images[0]
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)
                
            return img