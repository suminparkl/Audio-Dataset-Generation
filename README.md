# Sumin 

## 1. 설정 파일: `config.yaml`

`config.yaml`

```yaml
stem_dir: "data/musdb/train"             # 음악 스템 파일이 저장된 디렉토리
speech_dir: "data/LibriSpeech/dev-clean" # 음성 파일이 저장된 디렉토리
noise_dir: "data/noise/sound-bible"      # 노이즈 파일이 저장된 디렉토리
output_dir: "processed_data/train"       # 처리된 파일을 저장할 디렉토리
result_dir: "processed_data/results"     # 학습/검증/테스트 데이터 분할 결과를 저장할 디렉토리

speech_mix_ratio: 0.7                    # 음성과 무음의 비율 (1.0이면 음성 세그먼트 간 간격이 없음)
speech_gain: 1.0                         # 음성 볼륨 조정 비율 (1.0이면 변경 없음)

noise_mix_ratio: 0.5                     # 노이즈와 무음의 비율 (1.0이면 노이즈 세그먼트 간 간격이 없음)
noise_snr: 5                             # 노이즈 신호의 SNR 값 (단위: dB)
```
## 2. requirements.txt
## 3. gen_dataset.py

## 4. Dataset

```
LibriSpeech.dev-clean
musdb.train
musan.noise.sound-bible

# 추후에 google-drive link 추가 예정
```
