import torch
import open_clip
# import torch
# import torch.nn as nn
# import torch.nn.functional as F



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
        print("Model backbone parameters frozen (requires_grad = False) loaded.")
        
    return model.to(device), preprocess, tokenizer, device


class DualBranchAdapter(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=512*2):
        super().__init__()

        # image branch
        self.img_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, dim))

        # text branch
        self.txt_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, dim))

        # mul(D) + diff(D) = 2D
        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim * 2, hidden_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dim, 1))

    def forward(self, image_feature, text_feature):

        # image_feature: [B,D]
        # text_feature : [C,D]

        image = image_feature.unsqueeze(1)
        text  = text_feature.unsqueeze(0)

        # projection
        h_img = self.img_mlp(image)
        h_txt = self.txt_mlp(text)

        # normalize
        h_img = torch.nn.functional.normalize(h_img, dim=-1)
        h_txt = torch.nn.functional.normalize(h_txt, dim=-1)

        # expand
        h_img_expand = h_img.expand(-1           , h_txt.size(1), -1)
        h_txt_expand = h_txt.expand(h_img.size(0), -1           , -1)

        # interaction
        mul_feature = (h_img_expand * h_txt_expand)

        # # min-max features
        # min_feature = torch.minimum(
        #     h_img_expand,
        #     h_txt_expand
        # )

        # max_feature = torch.maximum(
        #     h_img_expand,
        #     h_txt_expand
        # )        
        
        # element-wise difference
        diff_feature = torch.abs(h_img_expand - h_txt_expand)   # [B,C,D]

        # fusion
        fused = torch.cat([mul_feature, diff_feature], dim=-1)  # [B,C,2D]

        # classifier
        logits = self.classifier(fused).squeeze(-1)

        return logits
    