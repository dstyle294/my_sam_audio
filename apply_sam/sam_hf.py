import os
from extractors import Birdset
from sam_preprocessor import sam_transform
from datasets import Audio, load_dataset

import csv

birdset_extractor = Birdset()

regions = ["HSN", "PER", "UHH", "SNE", "POW", "NES"]

ds = load_dataset("DBD-research-group/BirdSet", regions[1], trust_remote_code=True)

METADATA_FILE = "metadata.csv"

output_dir = "/home/s.dalal.334/mount/SAM/PER"
model = "sam-audio-base"
type = "32k"
split = "test"
num_clips = len(ds["test_5s"])

os.makedirs(output_dir, exist_ok=True)

os.makedirs(os.path.join(output_dir), exist_ok=True)
os.makedirs(os.path.join(output_dir, model), exist_ok=True)
os.makedirs(os.path.join(output_dir, model, type), exist_ok=True)
os.makedirs(os.path.join(output_dir, model, type, split), exist_ok=True)
# os.makedirs(os.path.join(output_dir, "no_sam"), exist_ok=True)

csv_path_sam = os.path.join(output_dir, model, type, split, METADATA_FILE)
# csv_path_no_sam = os.path.join(output_dir, "no_sam", METADATA_FILE)

column_names = ds["test_5s"].column_names

with open(csv_path_sam, mode='w', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)

  writer.writerow(["file_name"] + column_names[2:])

# with open(csv_path_no_sam, mode='w', newline='', encoding='utf-8') as f:
#   writer = csv.writer(f)

#   writer.writerow(column_names[1:])

ds["test_5s"] = ds["test_5s"].select(range(num_clips))

print(ds["test_5s"][0])

ds["test_5s"] = ds["test_5s"].map(sam_transform, batched=True, batch_size=4, load_from_cache_file=False, keep_in_memory=False, fn_kwargs={"prompt": "A bird chirping"})


# # set_transform is applied on-the-fly, while map is applied fully to save

# ads = ads.cast_column("audio", Audio(decode=False))

# ads.save_to_disk("/home/s.dalal.334/SAM/audio/after_sam_100")
