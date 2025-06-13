import torch
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        self.device = device

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token embedding
