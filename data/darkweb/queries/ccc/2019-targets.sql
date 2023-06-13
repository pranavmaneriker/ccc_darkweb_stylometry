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
		 	AND EXTRACT(YEAR FROM posts.created_on) = 2019
		 	GROUP BY creator
		) ccounts_t
	ON ccounts_t.creator = P.creator
	WHERE EXTRACT(YEAR FROM P.created_on) = 2019
		    AND EXTRACT(MONTH FROM P.created_on) = 12
			AND ccounts_t.num_posts >= 1
	GROUP BY P.creator, P.creator_id) r;

