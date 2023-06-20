Dataset Generation for Stylometry from CrimBB Dread
===================================================

Note: The instructions assume that the [CrimeBB](https://dl.acm.org/doi/10.1145/3178876.3186178) data from [CCC](https://www.cambridgecybercrime.uk/process.html) is loaded into local postgres DB.
You may need to supply redentials to this DB to run some queries.

## Queries on CrimeBB

Create a dataset
```bash
psql -h localhost -U postgres  crimebb_dread --tuples-only -o datasets/2018-train < queries/ccc/2018-train.sql
```

### Range of posts

```sql
SELECT MAX(created_on), MIN(created_on) FROM posts;
```

```jsx
"max"	"min"
"2020-01-09 16:09:34.926212-05"	"2018-02-15 10:47:00-05"
```

### Monthly counts

```sql
SELECT post_year, post_month, COUNT(*) FROM
	(SELECT EXTRACT(YEAR FROM created_on) AS post_year, EXTRACT(MONTH FROM created_on) AS post_month 
	 FROM posts) t1
	 GROUP BY t1.post_year, t1.post_month;
```

```jsx
"post_year"	"post_month"	"count"
2018	2	919
2018	3	6426
2018	4	7248
2018	5	707
2018	6	7333
2018	7	7411
2018	8	6878
2018	9	5642
2018	10	6037
2018	11	4957
2018	12	3811
2019	1	5307
2019	2	6013
2019	3	10812
2019	4	23702
2019	5	21441
2019	6	21527
2019	7	22565
2019	8	71055
2019	9	9728
2019	10	15283
2019	11	5554
2019	12	14941
2020	1	9299
```

### Extract an example dataset

```sql
SELECT array_agg(P.id) AS Id, 
	   array_agg(B.name) AS Board, 
	   array_agg(P.thread_id) AS Thread_id, 
	   array_agg(P.content) AS Content,  
	   array_agg(P.created_on) AS Created_on,
	   array_agg(P.updated_on) AS Updated_on,
	   creator_id, creator 
FROM posts P INNER JOIN boards B
ON P.board_id = B.id
GROUP BY creator, creator_id LIMIT 10;
```

### Check counts

```sql
SELECT COUNT(*) as num_posts,
  	EXTRACT(MONTH FROM created_on) as month, 
	EXTRACT(YEAR FROM created_on) as year,
	creator
FROM posts  GROUP BY creator, month, year;

```

```sql
SELECT COUNT(*) as num_posts,
	creator
FROM posts  GROUP BY creator;
```

### Extract actual dataset for each split

2018-train

```sql
SELECT row_to_json(r) FROM
(
	SELECT array_agg(P.id) AS id, 
		   array_agg(B.name) AS board, 
		   array_agg(P.thread_id) AS thread_id, 
		   array_agg(P.content) AS content,  
		   array_agg(P.created_on) AS created_on,
		   array_agg(P.updated_on) AS updated_on,
		   creator_id, P.creator AS creator
	FROM posts P INNER JOIN boards B 
	ON P.board_id = B.id
	INNER JOIN
		(SELECT COUNT(*) as num_posts,
				creator
			FROM posts  GROUP BY creator) ccounts
	ON ccounts.creator = P.creator
	WHERE EXTRACT(YEAR FROM P.created_on) = 2018
		    AND EXTRACT(MONTH FROM P.created_on) <= 10
			AND ccounts.num_posts > 1
	GROUP BY P.creator, P.creator_id
) r;
```

```sql
psql -h localhost -U postgres  crimebb_dread < queries/ccc/2018-train.sql > datasets/2018-train
```

2018-query

```sql
SELECT row_to_json(r) FROM
(SELECT array_agg(P.id) AS id, 
		   array_agg(B.name) AS board, 
		   array_agg(P.thread_id) AS thread_id, 
		   array_agg(P.content) AS content,  
		   array_agg(P.created_on) AS created_on,
		   array_agg(P.updated_on) AS updated_on,
		   creator_id, P.creator AS creator
	FROM posts P INNER JOIN boards B 
	ON P.board_id = B.id
	INNER JOIN
		(SELECT COUNT(*) as num_posts,
				creator
			FROM posts 
			WHERE EXTRACT(MONTH FROM posts.created_on) = 11
		 	AND EXTRACT(YEAR FROM posts.created_on) = 2018
			GROUP BY creator
		) ccounts_q
	ON ccounts_q.creator = P.creator
	INNER JOIN
		(SELECT COUNT(*) as num_posts,
				creator
			FROM posts
			WHERE EXTRACT(MONTH FROM posts.created_on) = 12
		 	AND EXTRACT(YEAR FROM posts.created_on) = 2018
		 	GROUP BY creator
		) ccounts_t
	ON ccounts_t.creator = P.creator
	WHERE EXTRACT(YEAR FROM P.created_on) = 2018
		    AND EXTRACT(MONTH FROM P.created_on) = 11
			AND ccounts_q.num_posts >= 1
			AND ccounts_t.num_posts >= 1
	GROUP BY P.creator, P.creator_id) r;

```

2018-targets

```sql
SELECT row_to_json(r) FROM
(SELECT array_agg(P.id) AS id, 
		   array_agg(B.name) AS board, 
		   array_agg(P.thread_id) AS thread_id, 
		   array_agg(P.content) AS content,  
		   array_agg(P.created_on) AS created_on,
		   array_agg(P.updated_on) AS updated_on,
		   creator_id, P.creator AS creator
	FROM posts P INNER JOIN boards B 
	ON P.board_id = B.id
	INNER JOIN
		(SELECT COUNT(*) as num_posts,
				creator
			FROM posts
			WHERE EXTRACT(MONTH FROM posts.created_on) = 12
		 	AND EXTRACT(YEAR FROM posts.created_on) = 2018
		 	GROUP BY creator
		) ccounts_t
	ON ccounts_t.creator = P.creator
	WHERE EXTRACT(YEAR FROM P.created_on) = 2018
		    AND EXTRACT(MONTH FROM P.created_on) = 12
			AND ccounts_t.num_posts >= 1
	GROUP BY P.creator, P.creator_id) r;
```
