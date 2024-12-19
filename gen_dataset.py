import os
import random
import yaml
import logging
import argparse
import librosa
import pandas as pd
import numpy as np
import stempeg
from pydub import AudioSegment
from typing import List, Tuple, Optional
from tqdm import tqdm

def setup_logging() -> None:
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def normalize_audio(audio: AudioSegment, target_dBFS: float) -> AudioSegment:
    if audio.dBFS == float('-inf'):
        logging.warning("Audio is silent, skipping normalization.")
        return audio
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)

def calculate_speech_gain(music_audio: AudioSegment, speech_audio: AudioSegment, smr: float) -> float:
    """
    Gain to apply to speech so that speech is `smr` dB above music level.
    If music is silent, just leave speech as is.
    """
    if music_audio.dBFS == float('-inf'):
        return 0.0
    if speech_audio.dBFS == float('-inf'):
        return 0.0
    gain = (music_audio.dBFS + smr) - speech_audio.dBFS
    return gain

def calculate_noise_gain(mixture_audio: AudioSegment, noise_audio: AudioSegment, snr: float) -> float:
    """
    Calculate gain for noise so that the SNR of (mixture_audio vs noise) is at a certain level.
    If mixture is silent, set noise to -20 dBFS as a fallback.
    """
    if mixture_audio.dBFS == float('-inf'):
        # If mixture is silent, set noise around -20 dBFS
        return -20.0 - noise_audio.dBFS
    if noise_audio.dBFS == float('-inf'):
        return 0.0
    gain = (mixture_audio.dBFS - snr) - noise_audio.dBFS
    return gain

def load_transcriptions( commonvoice_tsv: str = None, commonvoice_clips_dir: str = None) -> dict:
    transcriptions = {}

    if commonvoice_tsv and commonvoice_clips_dir:
        tsv_path = commonvoice_tsv
        if not os.path.exists(tsv_path):
            logging.warning(f"Transcription TSV file not found at {tsv_path}. Transcriptions will be empty.")
            return transcriptions
        df = pd.read_csv(tsv_path, sep='\t')
        for _, row in df.iterrows():
            audio_file = row['path'].strip() if isinstance(row['path'], str) else None
            sentence = row['sentence'].strip() if isinstance(row['sentence'], str) else ""

            if audio_file and sentence:
                audio_path = os.path.join(commonvoice_clips_dir, audio_file)
                if os.path.exists(audio_path):
                    transcriptions[audio_file] = sentence
                else:
                    logging.warning(f"Speech file {audio_path} does not exist.")
    return transcriptions

def segment_random_durations(total_duration_sec: float, min_dur: float = 20.0, max_dur: float = 30.0) -> List[float]:
    durations = []
    remaining = total_duration_sec
    while remaining > max_dur:
        seg = random.uniform(min_dur, max_dur)
        durations.append(seg)
        remaining -= seg
    if remaining > 0:
        if remaining < min_dur and len(durations) > 0:
            durations[-1] += remaining
        else:
            durations.append(remaining)
    return durations

def get_transcription(file_name: str, transcriptions: dict) -> str:
    return transcriptions.get(file_name, "")
def get_random_audio_segment(
    file_list: List[str],
    duration: float,
    normalize_target: float = -20.0,
    is_speech: bool = False,
    transcriptions: Optional[dict] = None
) -> Tuple[AudioSegment, str]:
    """
    If is_speech=True:
        Keep concatenating random speech files until the total duration meets or exceeds the required 'duration'.
        Ensure only files with transcriptions are selected.
        If adding a file causes an overflow, skip that file and stop the process.
    Else (for noise):
        Just pick one random file, and pad or truncate.
    """
    if not file_list:
        logging.warning("No files available. Returning silence.")
        return AudioSegment.silent(duration=int(duration * 1000)), ""

    # Filter files with valid transcriptions
    if is_speech and transcriptions:
        file_list = [f for f in file_list if os.path.basename(f) in transcriptions]
        if not file_list:
            logging.warning("No files with transcriptions available. Returning silence.")
            return AudioSegment.silent(duration=int(duration * 1000)), ""

    if not is_speech:
        # Non-speech: single file logic
        chosen_file = random.choice(file_list)
        seg = AudioSegment.from_file(chosen_file).set_channels(2)
        seg = normalize_audio(seg, normalize_target)
        if seg.duration_seconds > duration:
            seg = seg[:int(duration * 1000)]
        else:
            diff = duration * 1000 - len(seg)
            seg += AudioSegment.silent(duration=int(diff))
        return seg, os.path.basename(chosen_file)

    # Speech logic: concatenate multiple files if needed
    total_seg = AudioSegment.silent(duration=0)
    collected_trans = []
    required_ms = int(duration * 1000)
    current_length = 0

    while current_length < required_ms:
        if not file_list:
            # If no files left, pad silence
            needed = required_ms - current_length
            total_seg += AudioSegment.silent(duration=needed)
            current_length += needed
            break

        chosen_file = random.choice(file_list)
        seg = AudioSegment.from_file(chosen_file).set_channels(2)
        seg = normalize_audio(seg, normalize_target)
        file_basename = os.path.basename(chosen_file)
        trans_str = get_transcription(file_basename, transcriptions) if transcriptions else ""

        seg_ms = len(seg)
        remaining_ms = required_ms - current_length

        # Skip if adding this file would exceed the required duration
        if seg_ms > remaining_ms:
            break  # Stop adding files entirely

        # Add the file if it fits within the remaining duration
        total_seg += seg
        current_length += seg_ms
        if trans_str:
            collected_trans.append(trans_str)

    combined_trans = " ".join(collected_trans).strip()
    return total_seg, combined_trans



def process_segments(
    music_audio: AudioSegment,
    speech_files: List[str],
    noise_files: List[str],
    transcriptions: dict,
    segment_durations: List[float],
    min_smr: float, max_smr: float,
    min_snr: float, max_snr: float
) -> Tuple[AudioSegment, AudioSegment, AudioSegment, AudioSegment, str]:
    """
    For each segment, randomly choose one of the four scenarios:
    1) SPEECH_ONLY: speech + noise (no music)
    2) SPEECH+MUSIC: speech + music + noise
    3) MUSIC_ONLY: music + noise (no speech)
    4) SILENCE_ONLY: noise only (no speech, no music)

    Ensure that SPEECH+MUSIC occurs at least 3 times (or as many as possible based on segment count).

    Noise is always present, but scaled according to the chosen scenario.

    Returns: music_track, speech_track, noise_track, mix_track, combined_transcription
    """

    num_segments = len(segment_durations)
    num_speech_music = min(3, num_segments)  # at least 3 times if possible
    remaining_segments = num_segments - num_speech_music

    # Pre-assign scenarios
    scenarios = ["SPEECH+MUSIC"] * num_speech_music
    remaining_scenarios = random.choices(
        ["SPEECH_ONLY", "MUSIC_ONLY", "SILENCE_ONLY"],
        weights=[2, 3, 1],  # Adjust weights as needed
        k=remaining_segments
    )
    # scenarios = random.choices(
    # ["SPEECH+MUSIC", "SPEECH_ONLY", "MUSIC_ONLY", "SILENCE_ONLY"],
    # weights=[0.2767,  0.2248, 0.2498, 0.2488],  # 각 시나리오에 25% 확률 부여
    # k=num_segments
    # )
    scenarios += remaining_scenarios
    random.shuffle(scenarios)

    logging.info(f"Assigned scenarios: {scenarios}")

    total_duration = sum(segment_durations)
    total_ms = int(total_duration * 1000)

    # Prepare empty final tracks
    final_music = AudioSegment.silent(duration=total_ms, frame_rate=music_audio.frame_rate).set_channels(2)
    final_speech = AudioSegment.silent(duration=total_ms, frame_rate=music_audio.frame_rate).set_channels(2)
    final_noise = AudioSegment.silent(duration=total_ms, frame_rate=music_audio.frame_rate).set_channels(2)
    final_mix = AudioSegment.silent(duration=total_ms, frame_rate=music_audio.frame_rate).set_channels(2)

    combined_transcription = ""
    current_pos = 0

    for idx, (seg_dur, scenario) in enumerate(zip(segment_durations, scenarios)):
        seg_ms = int(seg_dur * 1000)

        logging.debug(f"Processing segment {idx+1}/{num_segments} with scenario: {scenario}")

        # Extract corresponding music segment
        music_seg = music_audio[current_pos:current_pos + seg_ms]
        if scenario not in ["SPEECH+MUSIC", "MUSIC_ONLY"]:
            music_seg = AudioSegment.silent(duration=seg_ms, frame_rate=music_audio.frame_rate).set_channels(2)

        # speech and noise segments
        speech_seg = AudioSegment.silent(duration=seg_ms, frame_rate=music_audio.frame_rate).set_channels(2)
        speech_trans = ""

        if scenario in ["SPEECH_ONLY", "SPEECH+MUSIC"]:
            seg_speech, speech_trans_str = get_random_audio_segment(
                speech_files, seg_dur, normalize_target=-20.0,
                is_speech=True, transcriptions=transcriptions
            )
            speech_seg = seg_speech
            speech_trans = speech_trans_str

        seg_noise, _ = get_random_audio_segment(
            noise_files, seg_dur, normalize_target=-20.0,
            is_speech=False, transcriptions=None
        )

        if scenario == "SPEECH+MUSIC":
            smr = random.uniform(min_smr, max_smr)
            gain_speech = calculate_speech_gain(music_seg, speech_seg, smr)
            speech_seg = speech_seg.apply_gain(gain_speech)
            logging.debug(f"Applied SMR of {smr} dB: Speech gain {gain_speech} dB")

        mixture = AudioSegment.silent(duration=seg_ms, frame_rate=music_audio.frame_rate).set_channels(2)
        mixture = mixture.overlay(music_seg).overlay(speech_seg)

        snr = random.uniform(min_snr, max_snr)
        gain_noise = calculate_noise_gain(mixture, seg_noise, snr)
        seg_noise = seg_noise.apply_gain(gain_noise)
        logging.debug(f"Applied SNR of {snr} dB: Noise gain {gain_noise} dB")

        segment_mix = mixture.overlay(seg_noise)

        final_music = final_music.overlay(music_seg, position=current_pos)
        final_speech = final_speech.overlay(speech_seg, position=current_pos)
        final_noise = final_noise.overlay(seg_noise, position=current_pos)
        final_mix = final_mix.overlay(segment_mix, position=current_pos)

        if speech_trans:
            combined_transcription += (speech_trans + " ")

        current_pos += seg_ms

    return final_music, final_speech, final_noise, final_mix, combined_transcription.strip()

def extract_music_stem(stem_path: str) -> Tuple[AudioSegment, float, int]:
    audio, rate = stempeg.read_stems(stem_path)
    music = audio[0]

    if music.ndim == 1:
        channels = 1
    else:
        channels = music.shape[1]
    music_audio = AudioSegment(
        np.array(music * 32767, dtype=np.int16).tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=channels
    ).set_channels(2)

    duration = music_audio.duration_seconds
    return music_audio, duration, rate

def create_spleeter_csv(
    stem_files: List[str], stem_dir: str, speech_dir: str, noise_dir: str,
    subset_dir: str, csv_path: str,
    min_smr: float = 20.0, max_smr: float = 30.0,
    min_snr: float = 20.0, max_snr: float = 30.0
) -> None:
    commonvoice_clips_dir = speech_dir
    commonvoice_tsv = os.path.join(os.path.dirname(commonvoice_clips_dir), "other.tsv")
    transcriptions = load_transcriptions(
        commonvoice_tsv=commonvoice_tsv,
        commonvoice_clips_dir=commonvoice_clips_dir
    )

    data = []
    os.makedirs(subset_dir, exist_ok=True)

    if not stem_files:
        logging.error(f"No stem files provided for {subset_dir}")
        return

    speech_files_full = [
        os.path.join(root, f)
        for root, _, files in os.walk(speech_dir)
        for f in files if f.lower().endswith('.wav')
    ]
    noise_files_full = [
        os.path.join(root, f)
        for root, _, files in os.walk(noise_dir)
        for f in files if f.lower().endswith('.wav')
    ]

    logging.info(f"Total speech files: {len(speech_files_full)}")
    logging.info(f"Total noise files: {len(noise_files_full)}")

    for i, stem_file in enumerate(tqdm(stem_files, desc=f"Processing {subset_dir} stems")):
        stem_path = os.path.join(stem_dir, stem_file)
        stem_name = stem_file.split(".stem")[0]

        try:
            music_audio, music_duration, rate = extract_music_stem(stem_path)
            segment_durations = segment_random_durations(music_duration, min_dur=20.0, max_dur=30.0)

            final_music, final_speech, final_noise, final_mix, combined_transcription = process_segments(
                music_audio=music_audio,
                speech_files=speech_files_full,
                noise_files=noise_files_full,
                transcriptions=transcriptions,
                segment_durations=segment_durations,
                min_smr=min_smr,
                max_smr=max_smr,
                min_snr=min_snr,
                max_snr=max_snr
            )

            speech_path = os.path.join(subset_dir, f"{stem_name}_speech.wav")
            music_path = os.path.join(subset_dir, f"{stem_name}_music.wav")
            noise_path = os.path.join(subset_dir, f"{stem_name}_others.wav")
            mix_path = os.path.join(subset_dir, f"{stem_name}_mix.wav")

            final_music.export(music_path, format="wav")
            final_speech.export(speech_path, format="wav")
            final_noise.export(noise_path, format="wav")
            final_mix.export(mix_path, format="wav")

            data.append({
                "stem_name": stem_name,
                "speech_path": speech_path,
                "music_path": music_path,
                "others_path": noise_path,
                "mix_path": mix_path,
                "duration": music_duration,
                "transcription": combined_transcription,
            })

            logging.info(f"Processed {stem_file} with transcription length: {len(combined_transcription)}")

        except Exception as e:
            logging.error(f"Error processing {stem_file}: {e}")
            continue

    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logging.info(f"CSV file saved to {csv_path}")
    else:
        logging.warning(f"No data to write to CSV for {subset_dir}.")

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Create Spleeter CSV with Enhanced Speech-Music Mixing")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--seed', type=int, help='Set random seed for reproducibility')

    args = parser.parse_args()
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)

    train_stem_dir = config['stem_dir_train']
    test_stem_dir = config['stem_dir_test']
    train_speech_dir = config['commonvoice_clips_dir_train']
    test_speech_dir = config['commonvoice_clips_dir_test']
    noise_dir = config['noise_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Output directory: {output_dir}")

    train_stem_files = [f for f in os.listdir(train_stem_dir) if f.endswith(".stem.mp4")]
    if not train_stem_files:
        logging.error(f"No train stem files found in {train_stem_dir}")
        return

    logging.info(f"Total train stem files found: {len(train_stem_files)}")
    random.shuffle(train_stem_files)

    total = len(train_stem_files)
    train_end = int(total * 0.8)

    train_files = train_stem_files[:train_end]
    val_files = train_stem_files[train_end:]

    logging.info(f"Train size: {len(train_files)}, Validation size: {len(val_files)}")




    test_stem_files = [f for f in os.listdir(test_stem_dir) if f.endswith(".stem.mp4")]
    if not test_stem_files:
        logging.error(f"No train stem files found in {test_stem_dir}")
        return

    logging.info(f"Total test stem files found: {len(test_stem_files)}")

    logging.info(f"Test size: {len(test_stem_files)}")


    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    min_smr = config.get('min_smr', 20.0)
    max_smr = config.get('max_smr', 30.0)
    min_snr = config.get('min_snr', 20.0)
    max_snr = config.get('max_snr', 30.0)

    train_csv_path = os.path.join(output_dir, "train.csv")
    create_spleeter_csv(
        stem_files=train_files,
        stem_dir=train_stem_dir,
        speech_dir=train_speech_dir,
        noise_dir=noise_dir,
        subset_dir=train_dir,
        csv_path=train_csv_path,
        min_smr=min_smr,
        max_smr=max_smr,
        min_snr=min_snr,
        max_snr=max_snr
    )

    validation_csv_path = os.path.join(output_dir, "validation.csv")
    create_spleeter_csv(
        stem_files=val_files,
        stem_dir=train_stem_dir,
        speech_dir=train_speech_dir,
        noise_dir=noise_dir,
        subset_dir=val_dir,
        csv_path=validation_csv_path,
        min_smr=min_smr,
        max_smr=max_smr,
        min_snr=min_snr,
        max_snr=max_snr
    )

    test_csv_path = os.path.join(output_dir, "test.csv")
    create_spleeter_csv(
        stem_files=test_stem_files,
        stem_dir=test_stem_dir,
        speech_dir=test_speech_dir,
        noise_dir=noise_dir,
        subset_dir=test_dir,
        csv_path=test_csv_path,
        min_smr=min_smr,
        max_smr=max_smr,
        min_snr=min_snr,
        max_snr=max_snr
    )

if __name__ == "__main__":
    main()

