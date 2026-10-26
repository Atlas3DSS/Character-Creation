import persona
import numpy as np
import pytest
from persona_literature_embeddings import aggregate,neighbors,initialize

def test_embedding_pooling_normalization_and_invalid_inputs(tmp_path):
 assert np.linalg.norm(aggregate([[1,0],[0,1]]))==pytest.approx(1)
 with pytest.raises(ValueError):aggregate([[float('nan'),0]])
 with pytest.raises(ValueError):aggregate([[0,0]])
 p=tmp_path/'db.sqlite';initialize(p)
 with pytest.raises(ValueError):neighbors(p,'u',100)
