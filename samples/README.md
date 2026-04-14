# Sample Audio Files

This directory holds example audio clips for the Gradio demo.

After training and evaluation, run the following to extract sample clips from the test set:

```python
import os
import soundfile as sf
from datasets import load_from_disk

ds = load_from_disk("processed_data/clips_3s")
test = ds["test"]

os.makedirs("samples", exist_ok=True)

# Find one sample per target accent
targets = {"american": None, "british": None, "indian_south": None}
for i in range(len(test)):
    label = test[i]["labels"].item()
    from config import ID2LABEL
    accent = ID2LABEL[label]
    if accent in targets and targets[accent] is None:
        targets[accent] = i

for accent, idx in targets.items():
    if idx is not None:
        audio = test[idx]["input_values"].numpy()
        sf.write(f"samples/{accent}_sample.wav", audio, 16000)
        print(f"Saved samples/{accent}_sample.wav")
```

Expected files:
- `american_sample.wav`
- `british_sample.wav`
- `indian_south_sample.wav`
