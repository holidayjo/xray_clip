"""STEP 2 -- the model and the contrastive objective.

This is the conceptual heart of the method, so the comments below spend most of their words on
WHY the objective works rather than on the code, which is short.

THE ARCHITECTURE (paper, Methods -> "Architecture")

    "The uninitialized architectures consist of a Vision Transformer, ViT-B/32, for the image
     encoder, and a Transformer for the text encoder. We use a pre-trained Vision Transformer
     that accepts images of resolution 224x224. The text encoder Transformer has a base size of
     63 million parameters, 12 layers and a width of 512 with 8 attention heads. ... maximum
     token length of 77. We use the same initialization scheme used in CLIP."

    "We initialized the self-supervised model using the ViT-B/32 and Transformer architectures
     with pre-trained weights from OpenAI's CLIP model."

So: two encoders, both starting from OpenAI CLIP weights. Nothing is invented -- the paper's
contribution is WHAT IT IS TRAINED ON (radiology reports), not a new architecture.

THE OBJECTIVE (paper, Methods -> "Training")

    "We train the model by maximizing the cosine similarity between image and text embeddings
     of all valid image-report pairs in the batch while minimizing the cosine similarity
     between the embeddings of incorrect pairings in the batch."

Concretely, for a batch of N pairs:

  1. encode all N images  -> I, an [N, 512] matrix, each row L2-normalised
  2. encode all N texts   -> T, an [N, 512] matrix, each row L2-normalised
  3. S = I @ T.T          -> an [N, N] matrix of cosine similarities

     S[i][j] = "how well does image i match text j?"

     The CORRECT pairs sit on the diagonal: S[0][0], S[1][1], ... The other N*N-N entries are
     wrong pairings, formed for free just by shuffling the batch against itself.

  4. Ask two classification questions and average their losses:

       row i:    "of these N texts, which belongs to image i?"   answer: i
       column j: "of these N images, which belongs to text j?"   answer: j

     Both are ordinary cross-entropy with the answer being the diagonal index. That is why
     `labels = arange(N)` appears below -- it is not a dataset label, it is just "the right
     answer is the one on the diagonal".

WHY THIS BUYS ZERO-SHOT CLASSIFICATION LATER

Nothing in the objective mentions categories. But to score the diagonal highest, the image
encoder is forced to place an image near text that DESCRIBES it. Once that geometry exists,
ANY sentence becomes a usable query -- including one the training data never contained, like
"No Cardiomegaly", or a category nobody annotated. The classifier is not learned; it is
constructed at inference time out of text. That is the whole trick.
"""
import torch
import open_clip


def load_clip(model_name="ViT-B-32", pretrained="openai", device=None, freeze=False):
    """Loads CLIP with OpenAI's pretrained weights -- the paper's exact starting point.

    Unlike the rest of this repo, the backbone here is TRAINABLE: CheXzero's whole method is to
    keep training these two encoders on domain image-text pairs. freeze=True is offered only
    for measuring the untrained baseline.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    model = model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded {model_name} / {pretrained} on {device}  "
          f"({sum(p.numel() for p in model.parameters()):,} params, {n_train:,} trainable)")
    return model, preprocess, tokenizer, device


def clip_contrastive_loss(image_features, text_features, logit_scale):
    """The symmetric InfoNCE loss used by CLIP, and by this paper.

    image_features : [N, D], already L2-normalised
    text_features  : [N, D], already L2-normalised
    logit_scale    : scalar temperature. CLIP LEARNS this rather than fixing it -- it controls
                     how sharply the model is required to separate the correct pair from the
                     rest. It is stored in the model as log(scale) and exponentiated, then
                     clamped to at most 100 to keep the softmax numerically stable.

    Returns (loss, logits_per_image) so callers can inspect the similarity matrix.
    """
    # [N, N] similarity matrix, scaled. Row i = image i against every text.
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text  = logits_per_image.t()

    # "the right answer is the diagonal" -- index i for row i.
    labels = torch.arange(logits_per_image.shape[0], device=image_features.device)

    loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)   # image -> text
    loss_t = torch.nn.functional.cross_entropy(logits_per_text,  labels)   # text  -> image
    return (loss_i + loss_t) / 2, logits_per_image


def encode_batch(model, images, texts, tokenizer, device, context_length=77):
    """One forward pass: images and raw caption strings -> two L2-normalised feature matrices.

    Normalising is what turns a dot product into a cosine similarity, which is what the loss
    above assumes. The paper truncates text to context_length - 2 tokens, reserving two slots
    for the [SOS] and [EOS] markers; open_clip's tokenizer already handles that truncation.
    """
    tokens = tokenizer(list(texts), context_length=context_length).to(device)

    image_features = model.encode_image(images.to(device))
    text_features  = model.encode_text(tokens)

    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    text_features  = torch.nn.functional.normalize(text_features,  dim=-1)
    return image_features, text_features


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from chexzero.data_coco import load_caption_pairs, CocoCaptionPairs, build_transform

    model, _, tokenizer, device = load_clip()
    model.eval()

    # ---- 1. what does the loss look like on random vs perfect features? ----
    N, D = 8, 512
    scale = model.logit_scale.exp().clamp(max=100).item()
    rand_i = torch.nn.functional.normalize(torch.randn(N, D), dim=-1).to(device)
    rand_t = torch.nn.functional.normalize(torch.randn(N, D), dim=-1).to(device)
    loss_rand, _ = clip_contrastive_loss(rand_i, rand_t, scale)
    loss_perf, _ = clip_contrastive_loss(rand_i, rand_i, scale)     # text == image = perfect
    import math
    print(f"\nloss with random pairing   : {loss_rand.item():.4f}   (chance is ln(N)={math.log(N):.4f})")
    print(f"loss with perfect alignment: {loss_perf.item():.6f}   (0 means the diagonal wins outright)")
    print(f"learned logit_scale (temperature): {scale:.2f}")

    # ---- 2. the real thing: a batch of COCO images against their own captions ----
    pairs = load_caption_pairs("val2017")[:6]
    ds = CocoCaptionPairs(pairs, build_transform(), caption_mode="first")
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    caps = [ds[i][1] for i in range(len(ds))]

    with torch.no_grad():
        I, T = encode_batch(model, imgs, caps, tokenizer, device)
        loss, logits = clip_contrastive_loss(I, T, scale)
    S = (I @ T.t()).cpu()

    print(f"\ncosine similarity matrix S[image][text] -- the DIAGONAL is the correct pairing:")
    print("        " + "".join(f"  txt{j}" for j in range(len(caps))))
    for i in range(len(caps)):
        row = "".join(f" {S[i][j]:+.2f}" for j in range(len(caps)))
        star = "  <- diagonal is the row max" if S[i].argmax().item() == i else "  <- MISMATCH"
        print(f"  img{i} {row}{star}")
    print(f"\ncontrastive loss on this real batch: {loss.item():.4f}")
    print(f"top-1 image->text retrieval accuracy: "
          f"{(S.argmax(dim=1) == torch.arange(len(caps))).float().mean().item()*100:.0f}%")
    print("\n(pretrained CLIP already solves COCO retrieval -- which is exactly why COCO can")
    print(" validate the code but cannot reproduce the paper's finding: there is no domain gap.)")
