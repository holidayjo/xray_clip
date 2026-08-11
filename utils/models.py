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


@torch.no_grad()
def encode_label_prompts(model, tokenizer, label_names, context_length, device, prompt_template):
    """Fills prompt_template (must contain '{label}') once per label and encodes the
    resulting prompts into normalized CLIP text features. prompt_template should come from
    the dataset config so train.py and val.py always encode identical prompts."""
    label_texts   = [prompt_template.format(label=name) for name in label_names]
    text_tokens   = tokenizer(label_texts, context_length=context_length).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return text_features


class DualBranchAdapter(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=256):
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
        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim * 2, hidden_dim), # 2D: since we concatenated.
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dim, 1))

    def forward(self, image_feature, text_feature):
        # image_feature: [B,D] -> (batch, 512).
        # text_feature : [C,D] -> (class, 512).
        image = image_feature.unsqueeze(1)
        text  = text_feature.unsqueeze(0)
        # projection
        h_img = self.img_mlp(image) # (B, 1, 512)
        h_txt = self.txt_mlp(text)  # (1, C, 512)
        # normalize
        h_img = torch.nn.functional.normalize(h_img, dim=-1) # (B, 1, 512), L2 norm through the 512 dim. 
        h_txt = torch.nn.functional.normalize(h_txt, dim=-1) # (1, C, 512)
        # expand
        h_img_expand = h_img.expand(-1           , h_txt.size(1), -1) # (B, C, 512) --> c identical copies of the image embedding for each class.
        h_txt_expand = h_txt.expand(h_img.size(0), -1           , -1) # (B, C, 512)
        # interaction
        mul_feature  = (h_img_expand * h_txt_expand) # [B,C,D]
        # element-wise difference
        diff_feature = torch.abs(h_img_expand - h_txt_expand)   # [B,C,D]
        # fusion
        fused = torch.cat([mul_feature, diff_feature], dim=-1)  # [B,C,2D]
        # classifier
        logits = self.classifier(fused).squeeze(-1) # torch.nn.Linear ignore the B and C dim. 
                                                    # It forces the last dim to be the input dim, 
                                                    # and outputs a single value for each B,C pair.
                                                    # Therefore, the output shape is [B,C] since it is squeezed.
        return logits # Now we are going to put this to the cross-entropy loss.
    


class ImageLinearProbe(torch.nn.Module):
    """Plain linear classifier on frozen CLIP image embeddings -- no text branch at all.
    Used to test how linearly-separable BiomedCLIP's frozen image representation already is for a given label, isolated from
    DualBranchAdapter mul/diff text-interaction scheme."""
    def __init__(self, dim=512, num_labels=1):
        super().__init__()
        self.classifier = torch.nn.Linear(dim, num_labels)

    def forward(self, image_feature):
        return self.classifier(image_feature)


class DualBranchAdapter_temp(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, use_branch_mlp=True):
        super().__init__()
        # The two branch MLPs are separately parameterised, so nothing constrains them to
        # apply the same transformation to each modality -- measured at init, they cut the
        # image/text cosine-similarity correlation to ~0.10, i.e. they largely discard the
        # alignment CLIP's contrastive pretraining provides. With use_branch_mlp=False the
        # mul/diff interaction is computed in CLIP's native shared space instead, at 1/3 the
        # parameter count.
        self.use_branch_mlp = use_branch_mlp
        if use_branch_mlp:
            self.img_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_dim, dim))
            self.txt_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_dim, dim))

        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim * 2, hidden_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dim, 1))

    def forward(self, image_feature, text_feature):
        image = image_feature.unsqueeze(1)   # [B, 1, D]
        text  = text_feature.unsqueeze(0)    # [1, C, D]

        h_img = image
        h_txt = text

        # Kept even when the MLPs are skipped: callers already pass normalized features, so
        # this is a no-op then, but it keeps the class correct if one ever forgets.
        h_img = torch.nn.functional.normalize(h_img, dim=-1)
        h_txt = torch.nn.functional.normalize(h_txt, dim=-1)

        h_img_expand = h_img.expand(-1, h_txt.size(1), -1)   # [B, C, D]
        h_txt_expand = h_txt.expand(h_img.size(0), -1, -1)   # [B, C, D]

        mul_feature  = h_img_expand * h_txt_expand
        diff_feature = torch.abs(h_img_expand - h_txt_expand)

        fused  = torch.cat([mul_feature, diff_feature], dim=-1)   # [B, C, 2D]
        logits = self.classifier(fused).squeeze(-1)               # [B, C]
        return logits
