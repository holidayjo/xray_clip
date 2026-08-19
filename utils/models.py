import torch
import open_clip
import torchvision
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class _ResNetFeatureExtractor(torch.nn.Module):
    """
    Wraps torchvision ResNet50 so it exposes .encode_image(), 
    making it a drop-in for the CLIP model everywhere in this repo -- build_embedding_cache, 
    lookup_embeddings and the image-only heads all work unchanged. 
    The final fc is replaced by Identity, 
    so the output is the 2048-d globally-pooled feature rather than 1000 ImageNet class logits.
    """
    
    def __init__(self, weights):
        super().__init__()
        net             = torchvision.models.resnet50(weights=weights)
        self.output_dim = net.fc.in_features          # 2048
        net.fc          = torch.nn.Identity()
        self.net        = net

    def encode_image(self, images):
        return self.net(images)


def load_resnet_feature_extractor(weights="IMAGENET1K_V2", device=None):
    """
    Loads an ImageNet-pretrained ResNet50 as a FROZEN feature extractor, 
    returning the same 4-tuple shape as load_clip_model: (model, preprocess, tokenizer, device). 
    tokenizer is None because there is no text branch -- 
    so this backbone only works with image-only heads (XGBoost, linear, mlp), never with DualBranchAdapter.

    preprocess comes from the weights' own transforms(), 
    so the normalisation matches what the network was trained with (ImageNet mean/std, resize 232 -> centre-crop 224).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w      = getattr(torchvision.models.ResNet50_Weights, weights)
    model  = _ResNetFeatureExtractor(w)
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()
    print(f"Loaded ImageNet ResNet50 ({weights}) as frozen feature extractor, dim={model.output_dim}")
    return model, w.transforms(), None, device


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
    def __init__(self, dim=512, hidden_dim=512, residual=True, skip_to_classifier=True):
        super().__init__()
        # residual=True wraps each branch MLP in a skip connection: h = x + mlp(x) instead of
        # h = mlp(x). The mul/diff interaction below is only meaningful if the image and text
        # branches stay in a shared space, but two separately-parameterised MLPs do not preserve
        # CLIP's alignment -- measured at init, plain MLPs cut the image/text similarity
        # correlation to 0.07, while the residual form keeps it at 0.66. The MLP then learns a
        # correction to the embedding rather than replacing it. Same parameter count either way.
        self.residual = residual

        # skip_to_classifier=True feeds the branch features PAST the interaction, straight into
        # the classifier: [mul, diff, h_img, h_txt] instead of [mul, diff]. mul and |diff| are
        # invariant to a global sign flip and to swapping the two branches, so image-only and
        # text-only signal is unrecoverable from them -- an image-only probe on these embeddings
        # scores respectably by itself, but the adapter cannot use that pathway without this
        # skip. h_txt is constant per label, so it also gives the shared classifier explicit
        # class conditioning. Costs dim*2 extra classifier inputs.
        self.skip_to_classifier = skip_to_classifier

        # image branch
        self.img_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, dim))
        # text branch
        self.txt_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, dim))

        # mul(D) + diff(D) = 2D, plus h_img(D) + h_txt(D) = 4D when skip_to_classifier
        n_parts = 4 if skip_to_classifier else 2
        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim * n_parts, hidden_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dim, 1))

    def forward(self, image_feature, text_feature):
        # image_feature: [B,D] -> (batch, 512).
        # text_feature : [C,D] -> (class, 512).
        image = image_feature.unsqueeze(1)
        text  = text_feature.unsqueeze(0)
        # projection (residual: the MLP learns a correction to CLIP's embedding)
        h_img = image + self.img_mlp(image) if self.residual else self.img_mlp(image)   # (B, 1, 512)
        h_txt = text  + self.txt_mlp(text)  if self.residual else self.txt_mlp(text)    # (1, C, 512)

        # normalize -- load-bearing with residual=True, since the sum changes the vector norm
        h_img = torch.nn.functional.normalize(h_img, dim=-1) # (B, 1, 512), L2 norm through the 512 dim.
        h_txt = torch.nn.functional.normalize(h_txt, dim=-1) # (1, C, 512)
        # expand
        h_img_expand = h_img.expand(-1           , h_txt.size(1), -1) # (B, C, 512) --> c identical copies of the image embedding for each class.
        h_txt_expand = h_txt.expand(h_img.size(0), -1           , -1) # (B, C, 512)
        # interaction
        mul_feature  = (h_img_expand * h_txt_expand) # [B,C,D]
        # element-wise difference
        diff_feature = torch.abs(h_img_expand - h_txt_expand)   # [B,C,D]

        # fusion -- optionally skipping the un-interacted branch features forward as well
        parts = [mul_feature, diff_feature]
        if self.skip_to_classifier:
            parts += [h_img_expand, h_txt_expand]
        fused = torch.cat(parts, dim=-1)  # [B,C,2D] or [B,C,4D]

        # classifier
        logits = self.classifier(fused).squeeze(-1) # torch.nn.Linear ignore the B and C dim.
                                                    # It forces the last dim to be the input dim,
                                                    # and outputs a single value for each B,C pair.
                                                    # Therefore, the output shape is [B,C] since it is squeezed.
        return logits # Now we are going to put this to the cross-entropy loss.



# class DualBranchAdapter(torch.nn.Module):
#     def __init__(self, dim=512, hidden_dim=512, residual=True):
#         super().__init__()
#         # residual=True wraps each branch MLP in a skip connection: h = x + mlp(x) instead of
#         # h = mlp(x). The mul/diff interaction below is only meaningful if the image and text
#         # branches stay in a shared space, but two separately-parameterised MLPs do not preserve
#         # CLIP's alignment -- measured at init, plain MLPs cut the image/text similarity
#         # correlation to 0.07, while the residual form keeps it at 0.66. The MLP then learns a
#         # correction to the embedding rather than replacing it. Same parameter count either way.
#         self.residual = residual
#         # image branch
#         self.img_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
#                                            torch.nn.ReLU(),
#                                            torch.nn.Linear(hidden_dim, dim))
#         # text branch
#         self.txt_mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim),
#                                            torch.nn.ReLU(),
#                                            torch.nn.Linear(hidden_dim, dim))
#         # mul(D) + diff(D) = 2D
#         self.classifier = torch.nn.Sequential(torch.nn.Linear(dim * 2, hidden_dim), # 2D: since we concatenated.
#                                               torch.nn.ReLU(),
#                                               torch.nn.Linear(hidden_dim, 1))

#     def forward(self, image_feature, text_feature):
#         # image_feature: [B,D] -> (batch, 512).
#         # text_feature : [C,D] -> (class, 512).
#         image = image_feature.unsqueeze(1)
#         text  = text_feature.unsqueeze(0)
#         # projection
#         # h_img = self.img_mlp(image) # (B, 1, 512)
#         # h_txt = self.txt_mlp(text)  # (1, C, 512)
#         h_img = image + self.img_mlp(image) if self.residual else self.img_mlp(image)   # (B, 1, 512)
#         h_txt = text  + self.txt_mlp(text)  if self.residual else self.txt_mlp(text)    # (1, C, 512)
        
        
#         # normalize
#         h_img = torch.nn.functional.normalize(h_img, dim=-1) # (B, 1, 512), L2 norm through the 512 dim. 
#         h_txt = torch.nn.functional.normalize(h_txt, dim=-1) # (1, C, 512)
#         # expand
#         h_img_expand = h_img.expand(-1           , h_txt.size(1), -1) # (B, C, 512) --> c identical copies of the image embedding for each class.
#         h_txt_expand = h_txt.expand(h_img.size(0), -1           , -1) # (B, C, 512)
#         # interaction
#         mul_feature  = (h_img_expand * h_txt_expand) # [B,C,D]
#         # element-wise difference
#         diff_feature = torch.abs(h_img_expand - h_txt_expand)   # [B,C,D]
#         # fusion
#         fused = torch.cat([mul_feature, diff_feature], dim=-1)  # [B,C,2D]
#         # classifier
#         logits = self.classifier(fused).squeeze(-1) # torch.nn.Linear ignore the B and C dim. 
#                                                     # It forces the last dim to be the input dim, 
#                                                     # and outputs a single value for each B,C pair.
#                                                     # Therefore, the output shape is [B,C] since it is squeezed.
#         return logits # Now we are going to put this to the cross-entropy loss.
    


class ImageLinearProbe(torch.nn.Module):
    """Plain linear classifier on frozen CLIP image embeddings -- no text branch at all.
    Used to test how linearly-separable BiomedCLIP's frozen image representation already is for a given label, isolated from
    DualBranchAdapter mul/diff text-interaction scheme."""
    def __init__(self, dim=512, num_labels=1):
        super().__init__()
        self.classifier = torch.nn.Linear(dim, num_labels)

    def forward(self, image_feature):
        return self.classifier(image_feature)


class DualBranchAdapter_simple(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=256):
        super().__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim*2, hidden_dim),
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
