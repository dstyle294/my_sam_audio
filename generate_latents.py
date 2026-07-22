"""
SAM Latent Extraction Pipeline

Loads preprocessed BirdSet data (HuggingFace Arrow format) from disk,
runs sam_model.get_target_latents() on each split, and saves the results
as a dictionary:
    {
        latents: [shape = E x C x (D x 25)]
        labels:  [shape = E x N_C]
    }

Usage:
    python extract_sam_latents.py \
        --train_path /path/to/train_dataset \
        --test_path  /path/to/test_dataset \
        --output_dir /path/to/output \
        --description "bird audio" \
        --batch_size 8
"""

import argparse
import os
import torch
import numpy as np
from datasets import load_from_disk, Audio
from sam_audio.processor import SAMAudioProcessor, batch_audio, mask_from_sizes, Batch
from sam_audio import SAMAudio
import torchaudio
import librosa, soundfile as sf

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", required=True, help="Path to saved train HF dataset")
# parser.add_argument("--test_path",  required=True, help="Path to saved test HF dataset")
parser.add_argument("--output_dir", required=True, help="Directory to save latent .pt files")
parser.add_argument("--model", required=True, help="Type of SAM Model to use")
parser.add_argument(
    "--description",
    default="bird audio",
    help="Text description passed to SAM for all samples (single string or path to .txt with one per line)",
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--audio_column", default="audio", help="HF dataset column containing audio")
parser.add_argument("--label_column", default="labels", help="HF dataset column containing one-hot labels")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIRDSET_ROOT = '/home/s.dalal.800/BirdSet'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_descriptions(description_arg: str, n: int) -> list[str]:
    """Return a list of n descriptions."""
    if os.path.isfile(description_arg):
        with open(description_arg) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == n:
            return lines
        if len(lines) == 1:
            return lines * n
        raise ValueError(
            f"Description file has {len(lines)} lines but dataset has {n} rows. "
            "Provide either 1 line (broadcast) or one per sample."
        )
    return [description_arg] * n

def load_audio(sample, min_len, max_len, sampling_rate):
    path = sample["filepath"]

    file_info = sf.info(path)
    sr = file_info.samplerate
    total_duration = file_info.duration
    
    if sample["detected_events"] is not None:
        start = sample["detected_events"][0]
        end = sample["detected_events"][1]
        event_duration = end - start
        
        if event_duration < min_len:
            extension = (min_len - event_duration) / 2
            
            # try to extend equally 
            new_start = max(0, start - extension)
            new_end = min(total_duration, end + extension)
            
            if new_start == 0:
                new_end = min(total_duration, new_end + (start - new_start))
            elif new_end == total_duration:
                new_start = max(0, new_start - (new_end - end))
            
            start, end = new_start, new_end

        if end - start > max_len:
            # if longer than max_len
            end = min(start + max_len, total_duration)
            if end - start > max_len:
                end = start + max_len
    else:
        start = sample["start_time"]
        end = sample["end_time"]



    start, end = int(start * sr), int(end * sr)
    audio, sr = sf.read(path, start=start, stop=end)
    
    target_len = int(max_len * sampling_rate)
    if len(audio) > target_len:
        # match BirdSet's "randomly extract fixed interval" behavior
        max_start = len(audio) - target_len
        offset = np.random.randint(0, max_start + 1)
        audio = audio[offset : offset + target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return {
        **{k: v for k, v in sample.items() if k != 'audio'},
        'audio': {
            'array': audio,
            'sampling_rate': sr,
            'path': path
        }
    }

def audio_entry_to_tensor(entry, target_sr: int) -> torch.Tensor:
    """
    Convert a HF audio dict or raw array to a mono float32 tensor at target_sr.
    HF audio columns decode to {'array': np.ndarray, 'sampling_rate': int, 'path': str}.
    """
    if isinstance(entry, dict):
        array = torch.tensor(entry["array"], dtype=torch.float32)
        sr = entry["sampling_rate"]
    elif isinstance(entry, np.ndarray):
        array = torch.tensor(entry, dtype=torch.float32)
        sr = target_sr
    elif isinstance(entry, torch.Tensor):
        array = entry.float()
        sr = target_sr
    else:
        raise TypeError(f"Unexpected audio type: {type(entry)}")

    if array.ndim == 1:
        array = array.unsqueeze(0)  # (1, T)

    if sr != target_sr:
        array = torchaudio.functional.resample(array, sr, target_sr)

    return array  # (C, T)


def build_batch(
    audio_tensors: list[torch.Tensor],
    descriptions: list[str],
    processor: SAMAudioProcessor,
    device: torch.device,
) -> Batch:
    """Build a Batch from a list of (C, T) audio tensors."""
    # batch_audio expects list of (1, T) or (2, T) tensors or file paths
    wavs_padded, wav_sizes = batch_audio(audio_tensors, processor.audio_sampling_rate)
    sizes = processor.wav_to_feature_idx(wav_sizes)
    audio_pad_mask = mask_from_sizes(sizes)

    batch = Batch(
        audios=wavs_padded.to(device),
        sizes=sizes.to(device),
        wav_sizes=wav_sizes.to(device),
        descriptions=descriptions,
        hop_length=processor.audio_hop_length,
        audio_sampling_rate=processor.audio_sampling_rate,
        anchors=None,
        audio_pad_mask=audio_pad_mask.to(device),
        masked_video=None,
    )
    return batch


def extract_latents(
    dataset,
    model,
    processor: SAMAudioProcessor,
    descriptions: list[str],
    batch_size: int,
    device: torch.device,
    audio_column: str,
    label_column: str,
) -> dict:
    """
    Iterate over dataset in batches, run get_target_latents, collect results.

    Returns:
        {
            "latents": list of tensors, each shape [E, C, D, T_i]
            "labels":  torch.Tensor of shape [E, N_C]
        }
    """
    dataset = dataset.select_columns(["filepath", audio_column, label_column])
    all_latents = []
    all_labels  = []
    n = len(dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_rows = dataset[start:end]

        # --- audio ---
        audio_tensors = [
            audio_entry_to_tensor(entry, processor.audio_sampling_rate)
            for entry in batch_rows[audio_column]
        ]

        # --- labels ---
        batch_labels = torch.tensor(
            batch_rows[label_column], dtype=torch.float32
        )  # [B, N_C]

        # --- descriptions ---
        batch_descs = descriptions[start:end]

        # --- build Batch and run model ---
        batch = build_batch(audio_tensors, batch_descs, processor, device)

        with torch.inference_mode():
            target_latents = model.get_target_latents(batch)

        latents = torch.stack([t.squeeze(0) for t in target_latents], dim=0)  # [B, C, T]
        all_latents.append(latents.cpu())
        all_labels.append(batch_labels)

        print(f"  Processed {end}/{n} samples", end="\r")

    print()
    return {
        "latents": torch.cat(all_latents, dim=0),  # [E, C, T]
        "labels":  torch.cat(all_labels,  dim=0),  # [E, N_C]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = args.model
    print(f"Using model = {model}")

    # --- Load processor (derives hop_length and sample_rate from model config) ---
    print(f">> Loading SAMAudioProcessor from {model}")
    processor = SAMAudioProcessor.from_pretrained(f"facebook/{model}")
    print(f"   sample_rate={processor.audio_sampling_rate}, hop_length={processor.audio_hop_length}")

    # --- Load model (caller is responsible for instantiation + weight loading) ---
    # Import here so the script is importable without sam_audio installed
    print(f">> Loading SAMAudio model {model}")
    model = SAMAudio.from_pretrained(f"facebook/{model}", proxies=None, resume_download=False).to(device).eval()
    print("   Model loaded.")

    splits = {
        "train": args.train_path,
    }


    for split_name, split_path in splits.items():
        print(f"\n>> Processing split: {split_name}  ({split_path})")
        dataset = load_from_disk(split_path)
        n = len(dataset)
        print(f"   {n} samples")

        descriptions = load_descriptions(args.description, n)

        dataset = dataset.map(lambda ex: {"filepath": os.path.join(BIRDSET_ROOT, ex["filepath"])})
        dataset = dataset.select(range(100))

        dataset = dataset.map(
            load_audio,
            fn_kwargs={
                "min_len": 5,
                "max_len": 5,
                "sampling_rate": processor.audio_sampling_rate,
            },
            num_proc=4,   # parallelize across CPU workers since sf.read is I/O bound
            desc="Loading audio",
        )

        # dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=processor.audio_sampling_rate, decode=True))

        print(dataset[0]['audio'])

        print(dataset.features)

        result = extract_latents(
            dataset=dataset,
            model=model,
            processor=processor,
            descriptions=descriptions,
            batch_size=args.batch_size,
            device=device,
            audio_column=args.audio_column,
            label_column=args.label_column,
        )

        out_path = os.path.join(args.output_dir, f"{split_name}_latents.pt")
        torch.save(result, out_path)
        print(f"   Saved {n} samples → {out_path}")
        print(f"   latents: list of {len(result['latents'])} tensors, "
              f"first shape = {result['latents'][0].shape}")
        print(f"   labels:  {result['labels'].shape}")

    print("\n>> Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)