import os
from argparse import ArgumentParser
import json

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

from luar_model import Transformer as LUARTransformer
from metrics import retrieval


transformer_path = "/exp/scale22/data_luar/pretrained_weights"
RANDOM_NUM = 42

class JSONLDataset(Dataset):
    def __init__(self, df, tokenizer, tokenizer_max_len, author_col, text_col,
                 author2idx, max_episode_len) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.author = author_col
        self.text = text_col
        self.author2idx = author2idx
        self.max_episode_len = max_episode_len
        self.max_tokenizer_len = tokenizer_max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        author = row[self.author]
        author_idx = self.author2idx[author]
        texts = row[self.text]
        if self.max_episode_len > 0:
            texts = texts[:self.max_episode_len]
        num_texts_per_episode = len(texts)
        tokenized_text = self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.max_tokenizer_len, return_tensors="pt")
        num_samples_per_author = 1
        # dummy example of what the inputs should look like, the forward pass can be re-written so 
        # that it doesn't depend in the "num_samples_per_author" and instead just evaluates on a (batch_size, num_texts, max_length)

        data = [tokenized_text["input_ids"], tokenized_text["attention_mask"]] 
        # (batch_size, num_samples_per_author, num_text_per_episode, max_length)
        # batching is done in the dataloader
        data[0] = data[0].reshape(1, num_samples_per_author, num_texts_per_episode, self.max_tokenizer_len)
        data[1] = data[1].reshape(1, num_samples_per_author, num_texts_per_episode, self.max_tokenizer_len)
        return data, author_idx
    

def run_evaluation(model, query_ds, target_ds, n_jobs,  is_cuda=False):
    def embed_ds(ds, name="query"):
        authors, embs = [], []
        with torch.no_grad():
            model.eval()
            for data, auth_idx in tqdm(ds, desc=f"Embedding {name}"):
                if is_cuda:
                    data[0] = data[0].to("cuda:0")
                    data[1] = data[1].to("cuda:0")
                emb = model(data)
                embs.append(emb["episode_embedding"].squeeze().cpu().numpy())
                authors.append(auth_idx)
            authors = np.array(authors)
            embs = np.array(embs)
        return embs, authors
    
    q_embs, q_authors = embed_ds(query_ds, "query")
    t_embs, t_authors = embed_ds(target_ds, "target")
    metrics, authorwise_results = retrieval(q_embs, q_authors, t_embs, t_authors, n_jobs=n_jobs) 
    print(metrics)
    return metrics, authorwise_results

if __name__ == "__main__":
    p = ArgumentParser('Run evaluation code with model')
    p.add_argument('--dataset_dir', type=str)
    p.add_argument('--tokenizer_path', type=str, default="../models/tokenizers/")
    p.add_argument("--checkpoint_path", type=str, default="../models/scale-luar/scale_mud.pt")
    p.add_argument('--output_file', type=str)
    p.add_argument('--query_sample_size', type=int)
    p.add_argument('--max_episode_length', type=int, default=-1)
    p.add_argument('--num_jobs', type=int, default=2)
    p.add_argument("--queries", type=str, default="test_queries.jsonl")
    p.add_argument("--targets", type=str, default="test_targets.jsonl")
    p.add_argument("--author_key", type=str, default="author")
    p.add_argument("--text_key", type=str, default="body")
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--tokens_per_comment", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", choices=["luar"], default="luar")
    args = p.parse_args()
    args_dict = vars(args)

    data_dir = args.dataset_dir
    output_file = os.path.abspath(args.output_file)
    query_sample_size = args.query_sample_size
    n_jobs = args.num_jobs
    max_episode_len = int(args.max_episode_length)
    queries = args.queries
    targets = args.targets
    author = args.author_key
    text = args.text_key
    checkpoint_path = args.checkpoint_path
    tokenizer_path = args.tokenizer_path
    tokens_per_comment = args.tokens_per_comment
    use_cuda = args.use_cuda
    model_type = args.model

    if model_type == "luar":
        model = LUARTransformer()

        # load weights
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        # for CUDA:
        if use_cuda:
            # TODO support multi gpu inference
            model.to(f"cuda:{args.gpu}")

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(tokenizer_path, "paraphrase-distilroberta-base-v1")
        )
    else:
        ValueError("Model %s not supported", model_type)

    # setup the datasets
    query_df = pd.read_json(os.path.join(data_dir, queries), orient="records", lines=True)
    print(query_df.columns)
    query_df = query_df[[author, text]]
    target_df = pd.read_json(os.path.join(data_dir, targets), orient="records", lines=True)
    target_df = target_df[[author, text]]

    if query_sample_size is not None:
        # quick check runs
        query_df = query_df.sample(n=query_sample_size, random_state=RANDOM_NUM)
        target_df = target_df[target_df[author].isin(set(query_df[author]))]
    
    all_authors = set(query_df[author]).union(set(target_df[author]))
    author2idx = {author: ind for ind, author in enumerate(all_authors)}
    args_ds = (tokenizer, tokens_per_comment, author, text, author2idx, max_episode_len)
    query_ds = JSONLDataset(query_df, *args_ds)
    target_ds = JSONLDataset(target_df, *args_ds)

    eval_results, authorwise_results_ind = run_evaluation(model, query_ds, target_ds, n_jobs, use_cuda)
    idx2auth = {ind: auth for auth, ind in author2idx.items()}
    authorwise_results = {idx2auth[ind]: rank for ind, rank in authorwise_results_ind}
    
    full_output = {
        "args": args_dict,
        "results": eval_results,
        "authorwise_results": authorwise_results
    }
    with open(output_file, 'a') as f:
        json.dump(full_output, f, indent=2)

    print(f'output_file={output_file}')

