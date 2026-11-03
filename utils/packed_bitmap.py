"""Build leaf payloads directly in the packed, little-endian storage format."""
import numpy as np


def _make_chunked_leaf_bitmaps(A_ow_all, obj_idx, row_chunk=16384):
    """Bound temporary arrays for large leaves; preserve structural membership."""
    count=obj_idx.size
    present=np.zeros(A_ow_all.shape[1],dtype=np.bool_)
    for begin in range(0,count,row_chunk):
        sub=A_ow_all[obj_idx[begin:begin+row_chunk]].tocsr()
        present[sub.indices]=True
    keywords=np.flatnonzero(present).astype(np.int64,copy=False)
    # A direct vocabulary lookup avoids an nnz-sized inverse permutation and
    # repeated binary searches while filling the packed rows.
    positions=np.empty(A_ow_all.shape[1],dtype=np.int32)
    positions[keywords]=np.arange(keywords.size,dtype=np.int32)
    bits=np.zeros((keywords.size,(count+7)//8),dtype=np.uint8)
    for begin in range(0,count,row_chunk):
        sub=A_ow_all[obj_idx[begin:begin+row_chunk]].tocsr()
        rows=np.repeat(np.arange(begin,begin+sub.shape[0],dtype=np.int64),np.diff(sub.indptr))
        np.bitwise_or.at(bits,(positions[sub.indices],rows>>3),(1<<(rows&7)).astype(np.uint8))
    return obj_idx,keywords,bits


def make_leaf_keyword_object_bitmaps(A_ow_all, obj_idx):
    obj_idx = np.ascontiguousarray(np.asarray(obj_idx, dtype=np.int64).reshape(-1))
    count = obj_idx.size
    if count>65536 and A_ow_all.nnz>8_000_000:
        return _make_chunked_leaf_bitmaps(A_ow_all,obj_idx)
    sub = A_ow_all[obj_idx].tocsr()
    # Use structural membership, matching the original implementation, including
    # stored zeros and duplicate entries in noncanonical CSR input.
    keywords, positions = np.unique(sub.indices, return_inverse=True)
    keywords = keywords.astype(np.int64, copy=False)
    bits = np.zeros((keywords.size, (count + 7) // 8), dtype=np.uint8)
    rows = np.repeat(np.arange(count, dtype=np.int64), np.diff(sub.indptr))
    # Multiple objects may share a byte: ordinary advanced-index |= loses updates.
    np.bitwise_or.at(bits, (positions, rows >> 3),
                     (1 << (rows & 7)).astype(np.uint8))
    return obj_idx, keywords, bits
