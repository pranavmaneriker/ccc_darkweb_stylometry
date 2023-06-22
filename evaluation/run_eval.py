import os
from argparse import ArgumentParser
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

from luar_model import Transformer as LUARTransformer
from metrics import retrieval


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

        data = [d.reshape(num_samples_per_author, num_texts_per_episode, self.max_tokenizer_len) for d in data]
        #data[0] = data[0].reshape(1, num_samples_per_author, num_texts_per_episode, self.max_tokenizer_len)
        #data[1] = data[1].reshape(1, num_samples_per_author, num_texts_per_episode, self.max_tokenizer_len)
        return data, torch.tensor([author_idx])
    
def collate_fn(batch):
        data, author = zip(*batch)

        author = torch.stack(author)

        # Minimum number of posts for an author history in batch
        #min_posts = min([d[0].shape[1] for d in data])
        # max number of episodes across all authors in batch
        max_posts = max([d[0].shape[1] for d in data])
        # If min_posts < episode length, need to subsample
        #if min_posts < 16:
        #    data = [torch.stack([f[:, :min_posts, :] for f in feature])
        #            for feature in zip(*data)]
        ## Otherwise, stack data as is
        #else:
        #    data = [torch.stack([f for f in feature])
        #            for feature in zip(*data)]
        
        #pad right all features to max epsiode len 
        data = [torch.stack([F.pad(f, (0, 0, 0, max_posts - f.shape[1]), "constant", 0) for f in feature])
                for feature in zip(*data)]
        return data, author

def run_evaluation(model, query_ds, target_ds, n_jobs,  batch_size=32, is_cuda=False, is_dp=False):
    def embed_ds(ds, name="query"):
        authors, embs = [], []
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=12)
        with torch.no_grad():
            model.eval()
            # for data, auth_idx in tqdm(ds, desc=f"Embedding {name}"):
            for data, auth_idx in tqdm(dl):
                if is_cuda:
                    data = [d.to("cuda:0") for d in data]
                emb = model(data)
                if emb["episode_embedding"].shape == (1, 512):
                    embs.append(emb["episode_embedding"].squeeze().cpu().numpy())
                    authors.append(auth_idx.squeeze().cpu().numpy())
                else:
                    embs.extend(emb["episode_embedding"].squeeze().cpu().numpy())
                    authors.extend(auth_idx.squeeze().cpu().numpy())

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
    p.add_argument("--use_dp", action="store_true", help="Use data parallel")
    p.add_argument("--tokens_per_comment", type=int, default=32)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--model", choices=["luar"], default="luar")
    p.add_argument("--from_lightning", action="store_true")
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
    batch_size = args.batch_size
    model_type = args.model
    use_dp = args.use_dp

    if model_type == "luar":
        model = LUARTransformer(args.tokenizer_path)

        # load weights
        state_dict = torch.load(checkpoint_path)
        if args.from_lightning:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        # for CUDA:
        if use_cuda:
            if use_dp:
                model = nn.DataParallel(model)
            model.to(f"cuda:0")


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

    eval_results, authorwise_results_ind = run_evaluation(model, query_ds, target_ds, n_jobs, batch_size, use_cuda, use_dp)
    idx2auth = {ind: auth for auth, ind in author2idx.items()}
    authorwise_results = {idx2auth[ind]: rank for ind, rank in authorwise_results_ind["ranks"]}
    topk_results = {idx2auth[idx]: [idx2auth[n_idx] for n_idx in n_idxs] 
                    for idx, n_idxs in authorwise_results_ind["nearest_topk"].items()}
    
    full_output = {
        "args": args_dict,
        "results": eval_results,
        "authorwise_results": authorwise_results,
        "topk_results": topk_results
    }
    with open(output_file, 'w') as f:
        json.dump(full_output, f, indent=2)

    print(f'output_file={output_file}')

