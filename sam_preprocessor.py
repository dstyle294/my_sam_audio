import os
import numpy as np
import csv

import torch
import torchaudio

import torchaudio.transforms as T


from sam_audio import SAMAudio, SAMAudioProcessor

# from .preprocessors import PreProcessorBase

output_dir = "/home/s.dalal.334/mount/SAM/PER"
model = "sam-audio-base"
type = "32k"
split = "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = SAMAudio.from_pretrained(f"facebook/{model}").to(device).eval()
processor = SAMAudioProcessor.from_pretrained(f"facebook/{model}")

output_dir = f"/{output_dir}/{model}/{type}/{split}"

METADATA_FILE = "metadata.csv"

original_sr = 48_000
new_sr = 32_000

resampler = T.Resample(original_sr, new_sr)

def sam_transform(batch, prompt="A bird chirping"):
    audios = [example['path'] for example in batch["audio"]]
    descriptions = [prompt for example in batch["audio"]]

    inputs = processor(audios=audios, descriptions=descriptions).to(device) # type: ignore

    with torch.inference_mode():
        results = sam_model.separate(inputs)

    for item_idx, example in enumerate(batch["audio"]):
        stripped_path = example['path'].split("/")[-1]
        
        processed_audio = results.target[item_idx].detach().cpu()

        resampled_waveform = resampler(processed_audio)

        filename_sam = os.path.join(output_dir, f"sam_{stripped_path}")
        
        torchaudio.save(filename_sam, 
                        resampled_waveform, 
                        sample_rate=new_sr)
    
        csv_path_sam = os.path.join(output_dir, METADATA_FILE)

        with open(csv_path_sam, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            row = [f"sam_{stripped_path}"] + [batch[col][item_idx] for col in batch if col not in ["audio", "filepath"]]

            writer.writerow(row)   

        del processed_audio
    
    del inputs, results

    # for item_idx, example in enumerate(batch["audio"]):
    #     inputs = processor(audios=[example['path']], descriptions=[prompt]).to(device) # type: ignore

    #     with torch.inference_mode():
    #         result = sam_model.separate(inputs)

    #     stripped_path = example['path'].split("/")[-1]
        
    #     processed_audio = result.target[0].detach().cpu()

    #     resampled_waveform = resampler(processed_audio)

    #     # print(f"Original audio: {processed_audio}")
    #     # print(f"New sample: {resampled_waveform}")

    #     # print(f"Original audio length: {processed_audio.shape}")
    #     # print(f"New audio length: {resampled_waveform.shape}")
        

    #     filename_sam = os.path.join(output_dir, f"sam_{stripped_path}")
        
    #     torchaudio.save(filename_sam, 
    #                     resampled_waveform, 
    #                     sample_rate=new_sr)
    
    #     csv_path_sam = os.path.join(output_dir, METADATA_FILE)

    #     with open(csv_path_sam, mode='a', newline='', encoding='utf-8') as f:
    #         writer = csv.writer(f)

    #         row = [f"sam_{stripped_path}"] + [batch[col][item_idx] for col in batch if col not in ["audio", "filepath"]]

    #         writer.writerow(row)    
    
    #     del inputs, result, processed_audio



class SegmentAudioPreprocessors():
    def __init__(
        self,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(self.device).eval()
        self.processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
        # super().__init__(name="SegmentAudioPreprocessor")

    def __call__(self, batch):
        new_audio = []
        new_labels = []
        
        for item_idx in range(len(batch["audio"])):
            
            inputs = self.processor(audios=[batch["audio"][item_idx]], descriptions=["Bird chirping"]).to(self.device) # type: ignore

            with torch.inference_mode():
                result = self.sam_model.separate(inputs)

            new_audio.append(result.target[0])
            new_labels.append(label)
    
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        return batch