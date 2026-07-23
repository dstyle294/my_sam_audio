## Motivation

## Workflow

BirdSet will automatically save a dataset if you instantiate it's data module. Per their [repository](https://github.com/DBD-research-group/BirdSet), here is how to save a dataset to disk (and later load it).

```python
from birdset.configs.datamodule_configs import DatasetConfig, LoadersConfig
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_from_disk

# initiate the data module
dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir='data_birdset/HSN', # specify your data directory!
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        n_workers=3,
        val_split=0.2,
        task="multilabel",
        classlimit=500, #limit of samples per class 
        eventlimit=5, #limit of events that are extracted for each sample
        sampling_rate=32000,
    ),
    loaders=LoadersConfig(), # only utilized in setup; default settings
    transforms=BirdSetTransformsWrapper() # set_transform in setup; default settings to spectrogram
)

# prepare the data
dm.prepare_data()

# manually load the complete prepared dataset (without any transforms). you have to cast the column with audio for decoding
ds = load_from_disk(dm.disk_save_path)
```

You can use this path and pass it as --train_path/--test_path accordingly to [generate_latents](../generate_latents.py)