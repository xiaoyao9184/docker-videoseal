# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory,
such as the `facebook/video_seal` model below.

The folder structure for `./cache/huggingface/hub/models--facebook--video_seal` is as follows.

```
.
├── blobs
│   └── 57ea132753a7545ed4891362d974d27a-10
├── refs
│   └── main
└── snapshots
    └── 8037ef59ba2b2ec8fb8b55298ff37b8ccddd078d
        └── checkpoint.pth -> ../../blobs/57ea132753a7545ed4891362d974d27a-10

5 directories, 3 files
```

and `./cache/huggingface/hub/models--facebook--audioseal` like this

```
.
├── blobs
│   ├── b23dc99a981b5f95fe82b143e195e717-10
│   └── e2b33855995852462549e5f93d028bd0-10
├── refs
│   └── main
└── snapshots
    └── 199c1793b46a37b682fb3367b8b2dcb443de9d72
        ├── detector_base.pth -> ../../blobs/e2b33855995852462549e5f93d028bd0-10
        └── generator_base.pth -> ../../blobs/b23dc99a981b5f95fe82b143e195e717-10

5 directories, 5 files
```

It will use
- `./cache/huggingface/hub/models--facebook--video_seal/snapshots/8037ef59ba2b2ec8fb8b55298ff37b8ccddd078d`
- `./cache/audioseal/0f195d476dd87ca1bd7b09e6`
- `./cache/audioseal/94c8df0b1d5ea8e45af4c884`

For more details, refer to [up@cpu-offline/docker-compose.yml](./../docker/up@cpu-offline/docker-compose.yml).


## Pre-download for offline mode

Running in online mode will automatically download the model.

install cli

```bash
pip install -U "huggingface_hub[cli]"
```

download model

```bash
huggingface-cli download --revision main --cache-dir ./cache/huggingface/hub facebook/video_seal checkpoint.pth
huggingface-cli download --revision main --cache-dir ./cache/huggingface/hub facebook/audioseal generator_base.pth
huggingface-cli download --revision main --cache-dir ./cache/huggingface/hub facebook/audioseal detector_base.pth
cp ./cache/huggingface/hub/models--facebook--audioseal/snapshots/199c1793b46a37b682fb3367b8b2dcb443de9d72/generator_base.pth  ./cache/audioseal/0f195d476dd87ca1bd7b09e6
cp ./cache/huggingface/hub/models--facebook--audioseal/snapshots/199c1793b46a37b682fb3367b8b2dcb443de9d72/detector_base.pth  ./cache/audioseal/94c8df0b1d5ea8e45af4c884
```