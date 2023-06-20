# %%
import os
from tqdm import tqdm
import json
import duckdb
from typing import List

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
temp_file = os.path.join("~/temp/temp")

all_cols = list(non_auth_columns)
all_cols.append(author_col)
# %%

tablename = fname.replace("-", "_")
table = duckdb.read_json(os.path.join(data_dir, fname))
conn = duckdb.register(tablename, table)
conn.execute("PRAGMA enable_progress_bar=true")


# %%

print("Creating table")
sel_cols = ",".join([f"list({col}) as {col}" for col in non_auth_columns])
print("Sorting")
conn.execute(f"PRAGMA memory_limit='90GB'")
duckdb.sql(f"COPY (SELECT {','.join(all_cols)} FROM {tablename} ORDER BY created_utc) TO '{temp_file}' (FORMAT JSON);")
print("Creating")
duckdb.sql(f"COPY (SELECT {sel_cols}, {author_col} FROM read_json_auto('{temp_file}')\
              GROUP BY {author_col} HAVING len(list(id)) <= {max_count} \
           ) TO '/data/ccc/data/reddit_{tablename}' (FORMAT JSON)", 
           connection=conn)

# %% [markdown]
# ***

# %%
#duckdb.sql("COPY (SELECT id, author, body, subreddit, created_utc from read_ndjson('/data/reddit/torrents/data/reddit/comments/extracted/RC_2019-12', auto_detect=true)) TO '/data/reddit/torrents/data/reddit/comments/extracted/RC_2019-12.parquet' (FORMAT parquet)")

# %%
def jsonl_gen(fname = "RC_2018-01",
              tablename= "RC_2019_01",
              author_col = "author",
              non_auth_columns = ["id", "body", "subreddit", "created_utc"],
              max_memory="90GB"):
    #table = duckdb.read_json(os.path.join(data_dir, fname))
    table = duckdb.read_parquet(os.path.join(data_dir, fname))
    conn = duckdb.register(tablename, table)
    conn.execute(f"PRAGMA memory_limit='{max_memory}'")
    conn.execute("PRAGMA enable_progress_bar=true")
    print("Sorting")
    duckdb.sql(f"COPY (SELECT * FROM {tablename} ORDER BY created_utc) TO '{temp_file}' (FORMAT JSON);", 
               connection=conn)
    print("Generating")
    sel_cols = ",".join([f"list({col}) as {col}" for col in non_auth_columns])
    all_rows = duckdb.sql(f"COPY (SELECT {sel_cols}, {author_col} FROM read_json_auto('{temp_file}') GROUP BY {author_col} \
                            HAVING len(list(id)) <= {max_count}) \
                           TO '/data/ccc/data/reddit_{tablename}' (FORMAT JSON)"
                          , connection=conn)

# %%
print("Creating Reddit 2019")
jsonl_gen(fname="RC_2019-12.parquet", tablename="RC_2019_12")

# %%



