import torch
import open_clip
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import multiprocessing
import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import login
login(token="*****")

path = r"D:\text-to-image-2M\data_512_2M"
def get_files(folder):
    Files = os.listdir(os.path.join(path, folder))
    Files = [[folder, x[:-5]] for x in Files if x.endswith('.json')]
    return Files
def get_text_prompts(File):
    json_path = os.path.join(path, File[0], File[1]+'.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    prompt = data['prompt']
    return [[File[0], File[1], prompt]]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai"
)
model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
@torch.no_grad()
def encode_texts(texts):
    tokens = tokenizer(texts).to(device)
    text_features = model.encode_text(tokens)
    normalized_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return normalized_features.cpu().numpy()

if __name__ == "__main__":
    files = []
    folders = os.listdir(path)
    with multiprocessing.Pool(processes=8) as pool:
        print("Loading Files ...")
        outputs = list(tqdm(pool.imap(get_files, folders)))
        for Files in outputs:
            files.extend(Files)
    print("Number of Files: ", len(files))
    data = []
    with multiprocessing.Pool(processes=8) as pool:
        print("Loading data ...")
        outputs = list(tqdm(pool.imap(get_text_prompts, files)))
        for Files in outputs:
            data.extend(Files)
    print("Number of prompts: ", len(data))

    BATCH_SIZE = 1000
    folders = []
    files = []
    captions_embeddings = []
    texts = []
    print("Encoding captions...")
    for i in tqdm(range(len(data))):
        folders.append(data[i][0])
        files.append(data[i][1])
        texts.append(data[i][2])
        if len(texts)==BATCH_SIZE or i== len(data) - 1:
            captions_embeddings.append(encode_texts(texts))
            texts = []
    captions_embeddings = np.concatenate(captions_embeddings, axis = 0)
    print(captions_embeddings.shape)
    np.savez("clip_text_embeddings.npz", 
            embeddings=captions_embeddings,
            files=np.array(files),
            folders=np.array(folders)
        )