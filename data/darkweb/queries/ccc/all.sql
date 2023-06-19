SELECT row_to_json(r) FROM
(SELECT 
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
GROUP BY creator, creator_id) r;