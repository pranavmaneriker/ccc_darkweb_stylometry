import json
import os
from tqdm import tqdm

import duckdb
from collections import namedtuple


TableConfig = namedtuple("TableConfig", "tname author body utc subforum")
max_count = 1500
min_count = 2
train_pc = 0.7

reddit_config = TableConfig("reddit", "author", "body", "created_utc", "subreddit")
cbb_config = TableConfig("dread", "creator_id", "content", "created_on", "board") 
# any crimebb dataset

# files containing jsonl grouped by author, sorted by creation time
dread_all = "/data/ccc/data/dread_all.jsonl" 
dread_out = "final/dread/"
thehub_all = "/data/ccc/data/thehub_all.jsonl"
thehub_out = "final/thehub/"

reddit_1_all = "/data/ccc/data/reddit_RC_2018_01"
reddit_1_sample = "/data/ccc/data/reddit_RC_2018_01_sample.jsonl"
reddit_1_out = "final/reddit_2018_sample/"

reddit_2_all = "/data/ccc/data/reddit_RC_2019_12"
reddit_2_sample = "/data/ccc/data/reddit_RC_2019_12_sample.jsonl"
reddit_2_out = "final/reddit_2019_sample/"


def two_part_split(df, query_path, target_path, config):
    c = config
    non_auth_cols = ["id", c.body, c.subforum, c.utc]
    with open(query_path, "w") as q:
        with open(target_path, "w") as t:
            for idx, r in tqdm(df.iterrows()):
                n_posts = len(r["id"])
                query = {col: r[col][:n_posts//2] for col in non_auth_cols}
                query[c.author] = r[c.author]
                query["author_idx"] = idx
                q.write(json.dumps(query))
                q.write("\n")

                target = {col: r[col][n_posts//2:] for col in non_auth_cols}
                target[c.author] = r[c.author]
                target["author_idx"] = idx
                t.write(json.dumps(target))
                t.write("\n")


def create_standardized_data(jsonl_path, table_name, output_train, output_query, output_target,
                              config, split_second=True):
    table = duckdb.read_json(jsonl_path)
    conn = duckdb.register(table_name, table)
    c = config

    split_point = duckdb.sql(f"SELECT quantile_cont({c.utc}, {train_pc}) as split_point FROM \
                           (SELECT unnest({c.utc}) as {c.utc}  FROM {table_name}) t", connection=conn).fetchone()[0]

    r_utc = split_point
    print("Chosen split {}".format(r_utc))
    print("Generating first half")
    duckdb.sql(f" COPY (\
                SELECT id[:r_idx] as id, {c.utc}[:r_idx] as {c.utc}, \
                    {c.body}[:r_idx] as {c.body}, {c.subforum}[:r_idx] as {c.subforum}, {c.author}  FROM \
                (SELECT \
                    (1 + list_max([idx for idx in range(len({c.utc})) \
                    if {c.utc}[idx] < {r_utc}])) as r_idx, \
                    id, {c.utc}, {c.body}, {c.subforum}, {c.author}\
                FROM {table_name}) t WHERE r_idx is NOT NULL AND r_idx >= {min_count}  AND len(id) < {max_count}) TO '{output_train}' (FORMAT JSON);")
    n_train = duckdb.sql(f"SELECT COUNT(*) FROM read_json_auto('{output_train}')").fetchone()[0]
    print("Num train: {}".format(n_train))
    
    q_t = duckdb.sql(f"SELECT id[r_idx:] as id, {c.utc}[r_idx:] as {c.utc}, \
                 {c.body}[r_idx:] as {c.body}, {c.subforum}[r_idx:] as {c.subforum}, {c.author}  FROM \
             (SELECT list_min([idx for idx in range(len({c.utc})) \
                 if {c.utc}[idx] >= {r_utc}]) as r_idx, \
                 id, {c.utc}, {c.body}, {c.subforum}, {c.author}\
              FROM {table_name}) t WHERE len(id) - r_idx + 1  > {min_count} AND len(id) < {max_count}").df()

    print("Generating second half")
    if split_second: 
        two_part_split(q_t, output_query, output_target, c)
    else:
        q_t.to_json(output_query, orient="records", lines=True)

def get_cbb_stats(cbb_train, cbb_query, cbb_target):
    akey = cbb_config.author
    n_rows, avg_len, min_len, max_len = \
    duckdb.sql(f"SELECT COUNT(DISTINCT({akey})), AVG(n_id), MIN(n_id), MAX(n_id) FROM \
           (SELECT {akey}, COUNT(id) as n_id FROM \
            (SELECT unnest(id) as id, {akey} FROM read_json_auto('{cbb_train}') UNION \
             SELECT unnest(id) as id, {akey} FROM read_json_auto('{cbb_query}') UNION \
            SELECT unnest(id) as id, {akey} FROM read_json_auto('{cbb_target}')) \
             GROUP BY {akey})").fetchone()
    return n_rows, avg_len, min_len, max_len

def sample_jsonl(jsonl_path, n_rows, min_len, max_len):
    sampled_df = duckdb.sql(f"SELECT * FROM \
                            (SELECT * FROM read_json_auto('{jsonl_path}') \
                            WHERE len(id) >= {min_len} AND len(id) <= {max_len}) t \
                            USING SAMPLE {n_rows} ROWS;").df()
    return sampled_df


def create_half_splits(main_jsonl, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    all_rows = duckdb.sql(f"SELECT * FROM read_json_auto('{main_jsonl}') \
                          WHERE len(id) >= {min_count} AND len(id) <= {max_count}").df()
    two_part_split(all_rows, os.path.join(output_dir, "queries.jsonl"),
                   os.path.join(output_dir, "targets.jsonl"), config)

if __name__ == "__main__":
    # dread
    print("Generating Dread splits")
    dtrain, dquery, dtarget = (os.path.join(dread_out, t) for t in ["train.jsonl", "test_queries.jsonl", "test_targets.jsonl"])
    os.makedirs(dread_out, exist_ok=True)
    create_standardized_data(dread_all, "dread", dtrain, dquery, dtarget, cbb_config)

    print("Generating TheHub splits")
    th_train, th_query, th_target = (os.path.join(thehub_out, t) for t in ["train.jsonl", "test_queries.jsonl", "test_targets.jsonl"])
    os.makedirs(thehub_out, exist_ok=True)
    create_standardized_data(thehub_all, "thehub", th_train, th_query, th_target, cbb_config)

    # create reddit samples
    print("Generating Reddit Samples")
    n_rows, avg_len, min_len, max_len = get_cbb_stats(dtrain, dquery, dtarget)
    print("Dread stats")
    print("n_rows: {}| avg_posts_per_user: {}| min_posts_per_user: {}| max_posts_per_user: {}".format(
          n_rows, avg_len, min_len, max_len))

    print("Sample 1")
    reddit_sample_1_df = sample_jsonl(reddit_1_all, n_rows, min_len, max_len)
    reddit_sample_1_df.to_json(reddit_1_sample, orient="records", lines=True)
    r_train, r_query, r_target = (os.path.join(reddit_1_out, t) for t in ["train.jsonl", "test_queries.jsonl", "test_targets.jsonl"])
    create_standardized_data(reddit_1_sample, "reddit_2018", 
                            r_train, r_query, r_target, reddit_config)

    print("Sample 2")
    reddit_sample_2_df = sample_jsonl(reddit_2_all, n_rows, min_len, max_len)
    reddit_sample_2_df.to_json(reddit_2_sample, orient="records", lines=True)

    r_train, r_query, r_target = (os.path.join(reddit_2_out, t) for t in ["train.jsonl", "test_queries.jsonl", "test_targets.jsonl"])
    create_standardized_data(reddit_2_sample, "reddit_2018", 
                            r_train, r_query, r_target, reddit_config)


    print("Large Half splits") 

    print("Dread")
    out_dir = "final/two_part/dread/"
    os.makedirs(out_dir, exist_ok=True)
    create_half_splits(dread_all, out_dir, cbb_config)

    print("TheHub")
    out_dir = "final/two_part/thehub/"
    os.makedirs(out_dir, exist_ok=True)
    create_half_splits(thehub_all, out_dir, cbb_config)

    print("Reddit 2018 sample")
    out_dir = "final/two_part/reddit_2018_sample"
    os.makedirs(out_dir, exist_ok=True)
    create_half_splits(reddit_1_sample, out_dir, reddit_config)

    print("Reddit 2019 sample")
    out_dir = "final/two_part/reddit_2019_sample"
    os.makedirs(out_dir, exist_ok=True)
    create_half_splits(reddit_2_sample, out_dir, reddit_config)


