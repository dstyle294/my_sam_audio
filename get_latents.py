# Birdset Imports
from datasets import Audio, load_from_disk, load_dataset
from extractors import Birdset

# SAM Imports
import torch
import sam_preprocessor

region = "HSN"

data = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True)

# Embedding settings
split = "test_5s"
num_clips = 100
save_dir = "/home/s.dalal.334/SAM/embeddings/"

clips_to_embed = data[split].select(range(num_clips))

all_embeddings = clips_to_embed.map(sam_preprocessor.get_latent, batched=True, batch_size=1, load_from_cache_file=False, keep_in_memory=False, fn_kwargs={"prompt": "A bird chirping"})

save_path = save_dir + "bird_latents_dataset.pt"

torch.save({
  "embeddings": all_embeddings,
  "metadata": {
    "feature_order": "mean_then_max",
    "dim": 256
  }
}, save_path)