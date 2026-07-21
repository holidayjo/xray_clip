import torch
import open_clip


def load_clip_model(model_name, freeze_backbone=True, device=None):
    """Loads an OpenCLIP model and preprocessor, optionally freezing weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Loading {model_name} onto {device}...")
    model, preprocess = open_clip.create_model_from_pretrained(model_name)
    tokenizer         = open_clip.get_tokenizer(model_name)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("Model backbone parameters frozen (requires_grad = False).")
        
    return model.to(device), preprocess, tokenizer, device
