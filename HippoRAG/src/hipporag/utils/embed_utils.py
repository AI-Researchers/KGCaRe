from typing import List
import os
import torch
from tqdm import tqdm

# torch.AcceleratorError was introduced in newer PyTorch versions alongside RuntimeError for CUDA OOM
_OOM_EXCEPTION_TYPES = (RuntimeError,)
try:
    _OOM_EXCEPTION_TYPES = (RuntimeError, torch.AcceleratorError)
except AttributeError:
    pass


def _is_cuda_oom(e: Exception, device: torch.device) -> bool:
    return device.type == "cuda" and "out of memory" in str(e).lower()


def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                 key_batch_size=10000):
    device_env = os.getenv("HIPPORAG_KNN_DEVICE", "auto").lower()
    if device_env == "cpu":
        device = torch.device("cpu")
    elif device_env == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tqdm.write(f"[embed_utils] retrieve_knn using device={device}")
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    Args:
        query_ids:
        key_ids:
        k: top-k
        query_batch_size:
        key_batch_size:

    Returns:

    """

    if len(key_vecs) == 0:
        return {}

    query_vecs = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"
    ):
        query_batch = query_batch.clone().detach()
        try:
            query_batch = query_batch.to(device)
        except _OOM_EXCEPTION_TYPES as e:
            if _is_cuda_oom(e, device):
                tqdm.write("[embed_utils] CUDA OOM moving query batch to device; retrying on CPU")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                # query_batch is still on CPU since .to(device) failed; no move needed
            else:
                raise

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            try:
                key_batch = key_batch.to(device)
            except _OOM_EXCEPTION_TYPES as e:
                if _is_cuda_oom(e, device):
                    tqdm.write("[embed_utils] CUDA OOM moving key batch to device; switching to CPU")
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    # key_batch is still on CPU since .to(device) failed; no move needed
                    query_batch = query_batch.cpu()
                else:
                    raise

            actual_key_batch_size = key_batch.size(0)

            try:
                similarity = torch.mm(query_batch, key_batch.T)
            except _OOM_EXCEPTION_TYPES as e:
                if _is_cuda_oom(e, device):
                    tqdm.write("[embed_utils] CUDA OOM during matrix multiply; switching to CPU")
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    query_batch = query_batch.cpu()
                    key_batch = key_batch.cpu()
                    similarity = torch.mm(query_batch, key_batch.T)
                else:
                    raise

            topk_sim_scores, topk_indices = torch.topk(similarity, min(k, actual_key_batch_size), dim=1, largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # end for each kb batch

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)), dim=1,
                                                               largest=True, sorted=True)
        final_topk_indices = final_topk_indices.cpu()
        final_topk_sim_scores = final_topk_sim_scores.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # end for each query batch

    return results