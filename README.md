# Dataset Preparation for Audio Processing

## 1. Configuration file: `config.yaml`

`config.yaml`

```yaml

stem_dir_train: "data/musdb/train"
stem_dir_test : "data/musdb/test"
noise_dir: "data/noise/sound-bible"
output_dir: "processed_data_curated_1"  # Base directory for processed data
commonvoice_clips_dir_train : "data/cv-corpus-17-curated/common_voice_curated_train"
commonvoice_clips_dir_test : "data/cv-corpus-17-curated/common_voice_curated_test"
min_smr : -3                       # min Speech to Music Ratio(SMR)(unit: dB)3.09
max_smr : 6                   # max Speech to Music Ratio(SMR)(unit: dB)
min_snr: 0                       # min Signal to Noise Ratio(SNR)(unit: dB) 2.57
max_snr: 10                       # max Signal to Noise Ratio(SNR)(unit: dB) 2.57
                              
                              # SNR = 20 dB: The signal power is 100 times greater than the noise power.
                              # SNR = 0 dB: The signal and noise have equal power.
                              # SNR < 0 dB: The noise power is greater than the signal power.

seed : 42                    # Random seed for data reproduction


```
## 2. requirements.txt

```
numpy==1.24.3
pandas==2.0.3
PyYAML==6.0.2
pydub==0.25.1
librosa==0.10.0
stempeg==0.2.3
soundfile==0.12.1
argparse==1.4.0
tqdm==4.67.0
```
## 3. gen_dataset.py

```
 python3 gen_dataset.py --config config.yaml
```

## 4. Dataset

```
##common_voice_17_0
##musdb.train
##musan.noise.sound-bible


data/
├── cv-corpus-17-curated         # Common Voice curated dataset
│   ├── common_voice_curated_train
│   │   ├── common_voice_en_12345.wav
│   │   ├── common_voice_en_67890.wav
│   │   └── ...
│   └── common_voice_curated_test
│       ├── common_voice_en_2737.wav
│       ├── common_voice_en_71783.wav
│       └── ...
├── musdb                       # Music stem dataset
│   ├── train
│   │   ├── music-001.stem.mp4
│   │   ├── music-002.stem.mp4
│   │   └── ...
│   └── test
│       ├── music-003.stem.mp4
│       ├── music-004.stem.mp4
│       └── ...
└── musan                       # Noise dataset
    ├── music                   # Not used here
    │   ├── music-001.wav
    │   ├── music-002.wav
    │   └── ...
    ├── noise                   # Background noise sound dataset
    │   ├── free-sound          # Free Sound source
    │   │   ├── noise-sound-001.wav
    │   │   ├── noise-sound-002.wav
    │   │   └── ...
    │   └── sound-bible         # Sound Bible source
    │       ├── noise-sound-bible-0071.wav
    │       ├── noise-sound-bible-0072.wav
    │       └── ...
    └── speech                  # Not used here
        ├── speech-001.wav
        ├── speech-002.wav
        └── ...

```

## 5. Prepocessed Dataset for Test

```

processed_data_curated/
├── train/
│   ├── Actions - Devil's Words_mix.wav
│   ├── Actions - Devil's Words_music.wav
│   ├── Actions - Devil's Words_speech.wav
│   ├── Actions - Devil's Words_others.wav
│   ├── ...
├── validation/
│   ├── AvaLuna - Waterduct_mix.wav
│   ├── AvaLuna - Waterduct_music.wav
│   ├── AvaLuna - Waterduct_speech.wav
│   ├── AvaLuna - Waterduct_others.wav
│   ├── ...
├── test/
│   ├── Arise - Run Run Run_mix.wav
│   ├── Arise - Run Run Run_mix.wav
│   ├── Arise - Run Run Run_mix.wav
│   ├── Arise - Run Run Run_mix.wav
│   ├── ...
├── train.csv
├── validation.csv
└── test.csv

## Following our dataset is created with the parameters from the config.yaml above.

```

## 6.Google Drive Links for Downloading Datasets


- [Download MUSDB](https://drive.google.com/file/d/15QMdtI17JFjKzPLIVEMZDBJMJef7PJsx/view?usp=sharing)
- [Download MUSAN Noise](https://drive.google.com/file/d/1r-rqnSzligtNrYloBX4hCl7lkCR12ZQ1/view?usp=sharing)
- [Download our preprocessed curated Dataset](https://drive.google.com/file/d/1E2tcYXM7e3HgUGVa7oH2ntoG0-VcQR9o/view?usp=sharing)




## 7.Acknowledgements

- Our dataset is built using the following sources:
  - [Common Voice v17.0](https://commonvoice.mozilla.org/en/datasets)
  - [MUSDB18](https://sigsep.github.io/datasets/musdb.html)
  - [MUSAN](http://www.openslr.org/17/)

- [2024.12.19] Thanks to the contributors of these datasets for providing valuable resources for audio processing research.
