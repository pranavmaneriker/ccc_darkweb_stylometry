# %%
import os
from tqdm.notebook import tqdm
import json
import duckdb

# %%
data_dir = "/data/reddit/torrents/data/reddit/comments/extracted/"

# %% [markdown]
# ***

# %%
fname = "RC_2018-01"
author_col = "author"
non_auth_columns = ["id", "body", "subreddit", "created_utc"]
min_count = 4
max_count = 2000

# %%

tablename = fname.replace("-", "_")
table = duckdb.read_json(os.path.join(data_dir, fname))
conn = duckdb.register(tablename, table)


# %%
sel_cols = ",".join([f"list({col}) as {col}" for col in non_auth_columns])
all_rows = duckdb.sql(f"COPY (SELECT {sel_cols}, {author_col} FROM {tablename} GROUP BY {author_col}\
                         HAVING len(list(id)) <= {max_count} \
                      ) TO '/data/ccc/data/reddit_{tablename}' (FORMAT JSON)"
                      ,connection=conn)

# %% [markdown]
# ***

# %%
#duckdb.sql("COPY (SELECT id, author, body, subreddit, created_utc from read_ndjson('/data/reddit/torrents/data/reddit/comments/extracted/RC_2019-12', auto_detect=true)) TO '/data/reddit/torrents/data/reddit/comments/extracted/RC_2019-12.parquet' (FORMAT parquet)")

# %%
def jsonl_gen(fname = "RC_2018-01",
              tablename= "RC_2019_01",
              author_col = "author",
              non_auth_columns = ["id", "body", "subreddit", "created_utc"],
              max_memory="30GB"):
    #table = duckdb.read_json(os.path.join(data_dir, fname))
    table = duckdb.read_parquet(os.path.join(data_dir, fname))
    conn = duckdb.register(tablename, table)
    conn.execute(f"PRAGMA memory_limit='{max_memory}'")
    sel_cols = ",".join([f"list({col}) as {col}" for col in non_auth_columns])
    all_rows = duckdb.sql(f"COPY (SELECT {sel_cols}, {author_col} FROM {tablename} GROUP BY {author_col} \
                            HAVING len(list(id)) <= 2000) \
                           TO '/data/ccc/data/reddit_{tablename}' (FORMAT JSON)"
                          , connection=conn)

# %%
jsonl_gen(fname="RC_2019-12.parquet", tablename="RC_2019_12")

# %%



