import os

cmd = """
python src/safm_stage/extract_safm_features.py \
--input-image test_images/karthik.jpg \
--checkpoint best.pt \
--device cuda
"""

os.system(cmd)