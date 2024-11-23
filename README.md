# Sumin 

## 1. 설정 파일: `config.yaml`

`config.yaml`

```yaml
stem_dir: "data/musdb/train"
speech_dir: "data/LibriSpeech/dev-clean"
noise_dir: "data/noise/sound-bible"
output_dir: "processed_data"  # Base directory for processed data
speech_mix_ratio: 0.7         # If the mix_ratio is 1, there are no intervals between speech segments (connected continuously)
speech_gain: 1.0              # Speech volume adjustment ratio (1.0 means no change)
noise_mix_ratio: 0.5          # If the mix_ratio is 1, there are no intervals between noise segments (connected continuously)
noise_snr: 30                 # The SNR value with the noise signal (unit: dB)
                              
                              # SNR = 20 dB: The signal power is 100 times greater than the noise power.
                              # SNR = 0 dB: The signal and noise have equal power.
                              # SNR < 0 dB: The noise power is greater than the signal power.

background_gain : 1        # Background volume adjustment ratio (1.0 means no change)
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
##LibriSpeech.dev-clean
##musdb.train
##musan.noise.sound-bible


data/
├── LibriSpeech                  # speech dataset
│   └── dev-clean
│       ├── 84
│       │   ├── 121123
│       │   │   ├── 84-121123-0000.flac
│       │   │   ├── 84-121123-0001.flac
│       │   │   └── ...
│       └── ...
├── musdb                        # music stem dataset
│   └── train
│       ├── music-001.stem.mp4
│       ├── music-002.stem.mp4
│       └── ...
└── musan                         # noise dataset
    ├── music                     # not used here
    │   ├── music-001.wav
    │   ├── music-002.wav
    │   └── ...
    ├── noise                    # background noise sound dataset
    │   ├── free-sound           # Free Sound source
    │   │   ├── noise-sound-001.wav
    │   │   ├── noise-sound-002.wav
    │   │   └── ...
    │   └── sound-bible          # Sound Bible source
    │       ├── noise-sound-bible-0071.wav
    │       ├── noise-sound-bible-0072.wav
    │       └── ...
    └── speech                   # not used here
        ├── speech-001.wav
        ├── speech-002.wav
        └── ...
```

## 5. Prepocessed Dataset for Test

```
##LibriSpeech.dev-clean
##musdb.train
##musan.noise.sound-bible


processed_data/
├── train/
│   ├── mix_Music Delta - Disco.wav
│   ├── vocals_Music Delta - Disco.wav
│   ├── backgrounds_Music Delta - Disco.wav
│   ├── speech_Music Delta - Disco.wav
│   ├── noise_Music Delta - Disco.wav
│   ├── mix_The Wrong'Uns - Rothko.wav
│   ├── ...
├── test/
│   ├── mix_James May - All Souls Moon.wav
│   ├── vocals_James May - All Souls Moon.wav
│   ├── backgrounds_James May - All Souls Moon.wav
│   ├── speech_James May - All Souls Moon.wav
│   ├── noise_James May - All Souls Moon.wav
│   ├── ...
├── train.csv
├── validation.csv
├── test.csv
└── total_result.csv

## This test dataset created with the parameters from the config.yaml above.

```

# Google Drive Links for Downloading Datasets

- [Download LibriSpeech (dev-clean)](https://drive.google.com/file/d/1pbecU-SD_o2lyCMafsSM4SFQ9lWkfHzB/view?usp=drive_link)
- [Download LibriSpeech (train-clean-100)](https://drive.google.com/file/d/1HBw50T374ECaWX6XYTY7S1g4SpDj_U91/view?usp=drive_link)
- [Download MUSDB](https://drive.google.com/file/d/15QMdtI17JFjKzPLIVEMZDBJMJef7PJsx/view?usp=drive_link)
- [Download MUSAN Noise](https://drive.google.com/file/d/1r-rqnSzligtNrYloBX4hCl7lkCR12ZQ1/view?usp=sharing)
- [Download Preprocessed_Dataset_For_Test_deprecated](https://drive.google.com/file/d/1jnVaSDRp6R_Oq238djobyieSFayTcNBs/view?usp=drive_link)
- [Download Preprocessed_Dataset_For_Test_0.3](https://drive.google.com/file/d/1PdstxraED3I6m6-ycoxFFOSQXFEfAyVC/view?usp=drive_link)
- [Download Preprocessed_Dataset_For_Test_0.5](https://drive.google.com/file/d/1-vmDBos9N38qpTuwR7PxA72fsONMd4yA/view?usp=drive_link)





