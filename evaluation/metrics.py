# Copyright 2021 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import math
import numpy as np

from operator import itemgetter
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from six.moves import xrange
from absl import logging
import faiss


def compute_EER(fpr, tpr, thresh):
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]


def compute_EER_v2(fpr, tpr, thresholds):  # y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    # fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    print(f"c_def: {c_def}")
    print(f"min_c_det: {min_c_det}")
    min_dcf = min_c_det / c_def
    return min_dcf


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds


def calculate_author_search_metrics(D, query_authors, target_authors, D_target_author_indices=None):
    if isinstance(query_authors, list):
        query_authors = np.array(query_authors)
    if isinstance(target_authors, list):
        target_authors = np.array(target_authors)
    
    assert D.shape[0] == query_authors.shape[0]
    if D_target_author_indices is not None:
        assert D.shape[0] == D_target_author_indices.shape[0]
        assert D.shape[1] == D_target_author_indices.shape[1]
    else:
        assert D.shape[1] == target_authors.shape[0]

    # Compute rank
    num_queries = query_authors.shape[0]
    ranks = np.zeros((num_queries), dtype=np.int32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
    
    for query_index in xrange(num_queries):
        author = query_authors[query_index]
        distances = D[query_index]
        indices_in_sorted_order = np.argsort(distances)  # *increasing*
        # sorted D indices -> sorted target author indices -> sorted target authors
        if D_target_author_indices is not None:
            labels_in_sorted_order = target_authors[D_target_author_indices[query_index,indices_in_sorted_order]]
        else:
            labels_in_sorted_order = target_authors[indices_in_sorted_order]
        rank = np.where(labels_in_sorted_order == author)[0]
        # if the author isn't in the top-k, then rank is very large?
        if len(rank) == 1:
            rank = rank[0] + 1.
        else:
            rank = 1_000_000 + 1.
        ranks[query_index] = rank
        reciprocal_rank = 1.0 / float(rank)
        reciprocal_ranks[query_index] = (reciprocal_rank)

    result = {
        'MRR': np.mean(reciprocal_ranks),
        'MR': np.mean(ranks),
        'min_rank': np.min(ranks),
        'max_rank': np.max(ranks),
        'median_rank': np.median(ranks),
        'recall@1': np.sum(np.less_equal(ranks,1)) / np.float32(num_queries),
        'recall@2': np.sum(np.less_equal(ranks,2)) / np.float32(num_queries),
        'recall@4': np.sum(np.less_equal(ranks,4)) / np.float32(num_queries),
        'recall@8': np.sum(np.less_equal(ranks,8)) / np.float32(num_queries),
        'recall@16': np.sum(np.less_equal(ranks,16)) / np.float32(num_queries),
        'recall@32': np.sum(np.less_equal(ranks,32)) / np.float32(num_queries),
        'recall@64': np.sum(np.less_equal(ranks,64)) / np.float32(num_queries),
        'num_queries': num_queries,
        'num_targets': target_authors.shape[0]
    }
    authorwise_results = list(zip(query_authors.tolist(), ranks.tolist()))
    # make sure it's 
    return {k: float(v) if not 'num_' in k else int(v)
            for k, v in result.items()}, authorwise_results


def calculate_author_id_metrics(scores, query_authors_labels, target_authors_labels):
    if isinstance(query_authors_labels, list):
        query_authors_labels = np.array(query_authors_labels)
    if isinstance(target_authors_labels, list):
        target_authors_labels = np.array(target_authors_labels)
    
    assert scores.shape[0] == query_authors_labels.shape[0]
    assert scores.shape[1] == target_authors_labels.shape[0]

    # setup labels matrix
    q_labels = query_authors_labels.reshape((len(query_authors_labels),))
    t_labels = target_authors_labels.reshape((len(target_authors_labels),))
    q_rep = np.repeat(np.expand_dims(q_labels, 1), len(t_labels), axis=1)
    t_rep = np.repeat(t_labels.reshape((1,-1)), len(q_labels), axis=0)
    labels = q_rep == t_rep
    labels = labels.astype(int)
    del q_rep, t_rep, q_labels, t_labels
    
    labels, scores = labels.ravel(), scores.ravel()
    
    fpr, tpr, thresh = roc_curve(labels, scores)
    
    auc_score = roc_auc_score(labels, scores)
    EER = compute_EER(fpr, tpr, thresh)
    EER_v2 = compute_EER_v2(fpr, tpr, thresh) # compute_EER_v2(labels, scores)
    # fnr = 1 - tpr

    # print("From Kaldi:")
    fnrs, fprs, thresh2 = ComputeErrorRates(scores, labels)

    minDCF = ComputeMinDcf(fnrs, fprs, thresh2)

    auc_key = 'AUC_ROC'
    eer_key = 'EER'
    minDCF_key ='minDCF'

    labels_counts = np.bincount(labels)

    return {
        auc_key : auc_score,
        eer_key : EER,
        eer_key + '_v2': EER_v2,
        minDCF_key : minDCF,
        'LABEL_0_COUNT': int(labels_counts[0]),
        'LABEL_1_COUNT': int(labels_counts[1])
    }


def retrieval(query_vectors, query_authors, target_vectors,
              target_authors, metric='cosine', distances=None,
              n_jobs=None):
    """
    Arguments:
      query_vectors: Numpy matrix of size (N, V) for N authors and V features.
      query_authors: Numpy array of size (N) containing author IDs.
      target_vectors: Numpy matrix of size (M, V) for M authors and V features.
      target_authors: Numpy array of size (M) containing author IDs.
      metric: Metric used to compare different embeddings.
      pairwise_distances: If `metric` is "precomputed"
      n_jobs: (Optional) Number of threads to use to compute pairwise distances.

    Returns:
      A dictionary where the keys are the names of metrics
      and the values contain the corresponding computed 
      metrics.

    """

    # optionally calculate scores with and without reranking
    if metric == 'precomputed':
        D = distances
    else:
        D = pairwise_distances(query_vectors, Y=target_vectors, metric=metric, n_jobs=n_jobs)

    return calculate_author_search_metrics(D, query_authors, target_authors)


def author_id_reranking(query_vectors, query_authors, target_vectors,
                        target_authors, query_body_strs, target_body_strs,
                        rerank_model, top_k: int,
                        metric='cosine', n_jobs=None, distances=None):
    # assume elements in the same indices are associated with each other
    assert len(query_vectors) == len(query_body_strs)
    assert len(target_vectors) == len(target_body_strs)
    assert query_vectors.shape[1] == target_vectors.shape[1]

    # make sure each 'body' is a string so it works with cross-encoder
    if type(query_body_strs[0]) == list:
        query_body_strs = [' '.join(ls) for ls in query_body_strs]
    if type(target_body_strs[0]) == list:
        target_body_strs = [' '.join(ls) for ls in target_body_strs]
    
    # get distances
    # if metric == 'precomputed':
    #     D = distances
    # else:
    #     D = pairwise_distances(query_vectors, Y=target_vectors, metric=metric, n_jobs=n_jobs)
    
    target_index = faiss.IndexFlatL2(target_vectors.shape[1])
    target_index.add(target_vectors)

    D = np.zeros(shape=(query_vectors.shape[0], top_k), dtype=np.float32)
    D_target_author_indices = np.zeros(shape=(query_vectors.shape[0], top_k), dtype=np.int32)
    for i, (q_vec, q_body_str) in enumerate(zip(query_vectors, query_body_strs)):
        # get the top-k targets associated with query i
        _, top_k_result = target_index.search(q_vec.reshape((1, -1)), top_k)
        top_k_indices = top_k_result.ravel()
        # make lists of queries to compare with
        sorted_t_body_strs = [target_body_strs[j] for j in top_k_indices]
        
        # TODO: rerank batch size in case of large re-rank size?
        rerank_dists = rerank_model.Compare([q_body_str for _ in sorted_t_body_strs], sorted_t_body_strs)
        # write distances and original target indices
        for j, (dist, target_i) in enumerate(zip(rerank_dists, top_k_indices)):
            D[i, j] = dist[0]  # because each output is a list of length 1
            D_target_author_indices[i, j] = target_i  # record which target index that distance corresponds to

    # calculate metrics with re-ranked data
    return calculate_author_search_metrics(D, query_authors, target_authors, D_target_author_indices=D_target_author_indices)

  
def author_linking(query_authors_embeddings, query_authors_labels,
                   target_authors_embeddings, target_authors_labels,
                   metric='cosine', n_jobs=None):
    """
    query_authors: a small subset of the million users training data, these are our known bad guys,
                   each of the queries must also appear in the target set.
                   these conditions are enforced on preprocessing and assumed to be true here.
    target_authors: a small subset of 5000 users from the target month, the query authors are included
                    in this set.
    test_target_authors: a small subset of 5000 users from the test target month, setting this input 
                         calls for calibrated scores and target authors must be from the dev target
                         month
    for each query, we have an array of distances, how far the query is from each of the targets

    the closest target (min val in the array) is our best guess for the target matching the query,
    and the distance value is a proxy for confidence. The closer the two are, the more confident we are.
    """
    
    # rows -> number of queries, # columns -> number of targets, these are unique pairwise distances
    logging.info("Computing pairwise distances...")
    if metric == 'linear':
      logging.info("Linear kernel: assuming pre-normalized embeddings")
      scores = linear_kernel(query_authors_embeddings, Y=target_authors_embeddings)
    elif metric == 'angular':
      logging.info("Angular distance: assuming pre-normalized embeddings")
      cos_sim = linear_kernel(query_authors_embeddings, Y=target_authors_embeddings)
      clip_cos_sim = np.clip(cos_sim, -1.0, 1.0)
      scores = 1.0 - np.arccos(clip_cos_sim) / math.pi
    else:
      D = pairwise_distances(query_authors_embeddings, Y=target_authors_embeddings, metric=metric, n_jobs=n_jobs)
      scores = -1 * D

    return calculate_author_id_metrics(scores, query_authors_labels, target_authors_labels)
