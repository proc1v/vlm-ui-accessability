import os
import sys
from transformers import AutoTokenizer

ANNOTATIONS_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data/screen2ax_test/annotations"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

token_counts = []
line_counts = []

for fname in sorted(os.listdir(ANNOTATIONS_DIR)):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(ANNOTATIONS_DIR, fname), "r") as f:
        text = f.read().strip()
    
    tokens = tokenizer.encode(text)
    token_counts.append(len(tokens))
    line_counts.append(len(text.split("\n")))

print(f"Samples: {len(token_counts)}")
print(f"")
print(f"Tokens:  min={min(token_counts)}  mean={sum(token_counts)//len(token_counts)}  max={max(token_counts)}")
print(f"Lines:   min={min(line_counts)}  mean={sum(line_counts)//len(line_counts)}  max={max(line_counts)}")
print(f"")
print(f"Percentiles:")
import numpy as np
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: {int(np.percentile(token_counts, p))} tokens / {int(np.percentile(line_counts, p))} lines")