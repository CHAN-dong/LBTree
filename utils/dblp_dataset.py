"""Load prepared DBLP CSR directly, preserving empty rows and object IDs."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix


def load_dblp_dataset(directory, dtype=np.float32):
    directory=Path(directory)
    metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    shape=tuple(metadata['csr']['shape']);nnz=metadata['csr']['nnz']
    indices=np.memmap(directory/'indices.u32',dtype='<u4',mode='r',shape=(nnz,))
    indptr=np.memmap(directory/'indptr.u64',dtype='<u8',mode='r',shape=(shape[0]+1,))
    # Signed indices are required by SciPy. Use int32 where all offsets fit.
    index_type=np.int32 if max(shape[0],shape[1],nnz)<2**31 else np.int64
    matrix=csr_matrix((np.ones(nnz,dtype=dtype),indices.astype(index_type),
                       indptr.astype(index_type)),shape=shape,copy=False)
    with (directory/'vocab.json').open(encoding='utf-8') as f: vocabulary=json.load(f)
    return matrix,vocabulary,metadata
