import torch
import torch.nn as nn
from torchvision import transforms
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class OpenSource2DGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Castom Layers
        self.style_mapper = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def text_to_image(self, prompt, steps=100):
        text = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        
        # Generation with special layers
        style_code = self.style_mapper(text_features.float())
        img = self.image_decoder(style_code.view(-1, 128, 1, 1))
        
        # Optimize image
        img = self.optimize_image(img, text_features, steps)
        
        # Convert to PIL Image
        return self.tensor_to_image(img)

    def optimize_image(self, init_img, text_features, steps):
        img = init_img.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([img], lr=0.05)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # CLIP similarity loss
            img_features = self.clip_model.encode_image(
                transforms.functional.resize(img, (224, 224))
            loss = 1 - torch.cosine_similarity(text_features, img_features)
            
            # Regulation
            loss += 0.1 * (img[:, :, :-1, :] - img[:, :, 1:, :]).pow(2).mean()
            loss += 0.1 * (img[:, :, :, :-1] - img[:, :, :, 1:]).pow(2).mean()
            
            loss.backward()
            optimizer.step()
        
        return img.detach()

    @staticmethod
    def tensor_to_image(tensor):
        """Convert tensor to PIL Image"""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return Image.fromarray(img.astype('uint8'))

def show_image(pil_img, prompt):
    """Show image with no save"""
    plt.imshow(pil_img)
    plt.axis('off')
    plt.title(prompt)
    plt.show()

if __name__ == "__main__"
    generator = OpenSource2DGenerator().to(generator.device)
    
    # Interactive generation
    while True:
        user_prompt = input(" Enter the promt (or 'exit'): ")
        if user_prompt.lower() == 'exit':
            break
        
        print("Generation...")
        image = generator.text_to_image(user_prompt)
        show_image(image, user_prompt)
