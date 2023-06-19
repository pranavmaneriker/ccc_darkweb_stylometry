import psycopg2
import json
from tqdm import tqdm
import psycopg2.extras

dbname = "crimebb_dread"
output_file = "all"

conn = psycopg2.connect(user="<add>", password="<add>", host="localhost", database=dbname)
cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cursor.execute("""SELECT
    array_agg(P.id) as id,
    array_agg(P.thread_id) as thread_id,
    array_agg(P.content) as content,
    array_agg(P.created_on) as created_on,
    P.creator as creator,
    P.creator_id as creator_id,
    array_agg(P.updated_on) as updated_on,
    array_agg(B.name) as board
FROM posts P INNER JOIN boards B
ON P.board_id = B.id
GROUP BY creator, creator_id;
""")

with open(output_file, "w") as f:
    for result in  tqdm(cursor):
        f.write(json.dumps(result, default=str))
        f.write("\n")

conn.close()