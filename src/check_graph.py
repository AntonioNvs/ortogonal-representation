import warnings
warnings.filterwarnings("ignore")
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
import torch

dataset = get_dataset("rel-f1")
db = dataset.get_db(upto_test_timestamp=False)
stype_proposal = get_stype_proposal(db)
for table_name, col_stypes in stype_proposal.items():
    for col_name, col_stype in col_stypes.items():
        if col_stype in [stype.text_embedded, stype.text_tokenized]:
            stype_proposal[table_name][col_name] = stype.categorical

graph_data, _ = make_pkey_fkey_graph(db, col_to_stype_dict=stype_proposal)

print("Edge types:")
for et in graph_data.edge_types:
    print(et)
