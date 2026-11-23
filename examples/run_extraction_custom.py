#!/usr/bin/env python3
# examples/run_extraction_custom.py
import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from run_genie_experiments import get_dataset
from pygip.models.attack.genie_model_extraction import GenieModelExtraction

ds = get_dataset("CA-HepTh", api_type='pyg')
extractor = GenieModelExtraction(ds, attack_node_fraction=0.1, model_path="examples/watermarked_model_demo.pth")
# increase epochs for surrogate training:
extractor.surrogate_epochs = 200
res = extractor.attack()
print("Extraction results:", res)
