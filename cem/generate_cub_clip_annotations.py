import clip
import numpy as np
import os
import torch

from PIL import Image

device = 'cpu'
if not os.path.exists('/homes/me466/data/CUB200/CUB_200_2011/clip_ViT-B_32_zero_shot_attrs'):
    os.makedirs('/homes/me466/data/CUB200/CUB_200_2011/clip_ViT-B_32_zero_shot_attrs')

count = 0
for subdir, dirs, files in os.walk('/homes/me466/data/CUB200/CUB_200_2011/images/'):
    for file in files:
        if not file.endswith(".jpg"):
            continue
        count += 1

current_count = 0
embeddings_file = "/homes/me466/data/CUB200/cub_ViT-B_32_concept_embeddings.npy"
clip_concept_embeddings = torch.FloatTensor(np.load(embeddings_file))
for subdir, dirs, files in os.walk('/homes/me466/data/CUB200/CUB_200_2011/images/'):
    for file in files:
        if not file.endswith(".jpg"):
            continue
        image_filename = os.path.join(subdir, file)
        dirname = os.path.dirname(image_filename)
        emb_directory = dirname.replace('/images/', '/clip_ViT-B_32_zero_shot_attrs/')
        if not os.path.exists(emb_directory):
            os.makedirs(emb_directory)

        attributes_filename = os.path.join(emb_directory, file.replace('jpg', 'npy'))
        # if os.path.exists(attributes_filename):
        #     # Then we will ignore this guy as we already generated it!
        #     current_count += 1
        #     continue
        emb_path = os.path.join(dirname.replace('/images/', f'/clip_ViT-B_32_embeddings/'), file.replace('.jpg', '.npy'))
        img_clip_emb = torch.FloatTensor(np.load(emb_path))
        img_clip_emb = img_clip_emb / img_clip_emb.norm(dim=-1, keepdim=True)

        attr_label = np.zeros((clip_concept_embeddings.shape[0],))
        for concept_idx in range(clip_concept_embeddings.shape[0]):
            pos_emb = clip_concept_embeddings[concept_idx, 0, :]
            pos_emb = pos_emb / pos_emb.norm(dim=-1, keepdim=True)

            neg_emb = clip_concept_embeddings[concept_idx, 1, :]
            neg_emb = neg_emb / neg_emb.norm(dim=-1, keepdim=True)

            score_pos = (pos_emb @ img_clip_emb.t()).unsqueeze(-1)
            score_neg = (neg_emb @ img_clip_emb.t()).unsqueeze(-1)
            cos_probs = (100 * torch.concat([score_neg, score_pos], dim=0)).softmax(dim=0)
            attr_label[concept_idx] = cos_probs[1] #int(cos_probs[1] >= 0.5)

        current_count += 1
        print(f"[{100 * current_count/count:.2f}%] Saving embedding for", image_filename, "to", attributes_filename)
        with open(attributes_filename, 'wb') as f:
            np.save(f, attr_label)
