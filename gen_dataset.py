import os
import random
import soundfile as sf
import pandas as pd
import numpy as np
import stempeg
from pydub import AudioSegment
import librosa
import argparse
import yaml
import logging
from typing import List, Optional
from tqdm import tqdm
import shutil

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
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_train_val_test_split(stem_files: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
    """
    Split the list of stem files into train, validation, and test sets.

    Args:
        stem_files (List[str]): List of stem files.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.

    Returns:
        tuple: (train_files, val_files, test_files)
    """
    total = len(stem_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = stem_files[:train_end]
    val_files = stem_files[train_end:val_end]
    test_files = stem_files[val_end:]

    logging.info(f"Total files: {total}")
    logging.info(f"Train size: {len(train_files)}, Validation size: {len(val_files)}, Test size: {len(test_files)}")

    return train_files, val_files, test_files

def calculate_noise_gain(clean_audio: AudioSegment, noise_audio: AudioSegment, snr: float) -> float:
    """
    Calculate the gain needed for the noise audio to achieve the desired SNR.

    Args:
        clean_audio (AudioSegment): The clean audio segment.
        noise_audio (AudioSegment): The noise audio segment.
        snr (float): The desired signal-to-noise ratio in dB.

    Returns:
        float: The gain in dB to apply to the noise audio.
    """
    clean_rms = clean_audio.rms
    noise_rms = noise_audio.rms
    desired_noise_rms = clean_rms / (10 ** (snr / 20))
    if noise_rms == 0:
        return -float('inf')  # Handle silence in noise
    gain = 20 * np.log10(desired_noise_rms / noise_rms)


    # SNR = 20 dB: The signal power is 100 times greater than the noise power.
    # SNR = 0 dB: The signal and noise have equal power.
    # SNR < 0 dB: The noise power is greater than the signal power.

    return gain

def extract_and_save_stems(stem_path: str, output_dir: str) -> tuple:
    """
    Extract vocals and backgrounds from a stem file and save them.

    Args:
        stem_path (str): Path to the stem file.
        output_dir (str): Directory to save the outputs.

    Returns:
        tuple: Paths to the vocal and background audio files, duration, and sample rate.
    """
    stem_name = os.path.basename(stem_path).split(".stem")[0]
    vocal_path = os.path.join(output_dir, f"vocals_{stem_name}.wav")
    background_path = os.path.join(output_dir, f"backgrounds_{stem_name}.wav")
    # origin_music_path = os.path.join(output_dir, f"origin_music_{stem_name}.wav")

    # Extract stems 
    audio, rate = stempeg.read_stems(stem_path)
    vocals = audio[4]
    backgrounds = np.sum(audio[1:4], axis=0)
    # origin_music = audio[0]

    # 0 - The mixture,
    # 1 - The drums,
    # 2 - The bass,
    # 3 - The rest of the accompaniment,
    # 4 - The vocals.

    mixture_mono = np.mean(audio[0], axis=1)
    trimmed_indices = librosa.effects.trim(mixture_mono, top_db=20)[1]
    start_idx, end_idx = trimmed_indices
    vocals_trimmed = vocals[start_idx:end_idx, :]
    backgrounds_trimmed = backgrounds[start_idx:end_idx, :]

    # Save audio files
    sf.write(vocal_path, vocals_trimmed, rate)
    sf.write(background_path, backgrounds_trimmed, rate)
    # sf.write(origin_music_path, origin_music, rate)

    vocal_duration = (end_idx - start_idx) / rate

    return vocal_path, background_path, vocal_duration, rate

def process_audio_data(
    audio_dir: str, vocal_duration: float, rate: int, stem_name: str,
    mix_ratio: float, output_dir: str, audio_type: str,
    speech_gain: Optional[float] = None, snr: Optional[float] = None,
    reference_audio_path: Optional[str] = None
) -> Optional[str]:
    """
    Process audio data (speech or noise) and save the combined audio as stereo.

    Args:
        audio_dir (str): Directory containing audio files.
        vocal_duration (float): Duration of the vocal audio.
        rate (int): Sample rate.
        stem_name (str): Name of the stem file.
        mix_ratio (float): Ratio of the audio to mix.
        output_dir (str): Directory to save outputs.
        audio_type (str): Type of audio ('speech' or 'noise').
        background_gain (Optional[float]): Gain to apply to audio (for background).
        snr (Optional[float]): Desired signal-to-noise ratio (for noise).
        reference_audio_path (Optional[str]): Path to reference audio (for noise SNR calculation).

    Returns:
        Optional[str]: Path to the processed audio file or None if not processed.
    """
    if not audio_dir or mix_ratio <= 0:
        return None

    total_audio_duration = vocal_duration * mix_ratio
    total_silence_duration = vocal_duration * (1 - mix_ratio)
    audio_segments = []

    file_extension = ".flac" if audio_type == "speech" else ".wav"
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(audio_dir)
        for file in files if file.endswith(file_extension)
    ]
    if not audio_files:
        logging.error(f"No {audio_type} files found in {audio_dir}")
        return None
    
    # for reproduction remove randomness if you want, add below line

    random.shuffle(audio_files)

    current_duration = 0
    while current_duration < vocal_duration:
        if total_audio_duration > 0:
            if not audio_files:
                logging.warning(f"Not enough {audio_type} files.")
                break
            audio_file = audio_files.pop()
            audio_segment = AudioSegment.from_file(audio_file)
            audio_segment = audio_segment.set_channels(2)  # Ensure stereo
            segment_length = min(
                audio_segment.duration_seconds,
                total_audio_duration,
                vocal_duration - current_duration
            )
            audio_segment = audio_segment[:segment_length * 1000]
            audio_segments.append(audio_segment)

            current_duration += segment_length
            total_audio_duration -= segment_length
        else:
            break

        if current_duration < vocal_duration:
            silence_length = random.uniform(3, 5)
            silence_length = min(
                silence_length,
                total_silence_duration,
                vocal_duration - current_duration
            )
            silence = AudioSegment.silent(duration=silence_length * 1000).set_channels(2)  # Ensure stereo
            audio_segments.append(silence)

            current_duration += silence_length
            total_silence_duration -= silence_length

    if current_duration < vocal_duration:
        remaining_silence = vocal_duration - current_duration
        silence = AudioSegment.silent(duration=remaining_silence * 1000).set_channels(2)  # Ensure stereo
        audio_segments.append(silence)
        current_duration += remaining_silence

    combined_audio = sum(audio_segments)
    combined_audio = combined_audio.set_frame_rate(rate)
    combined_audio = combined_audio.set_channels(2)  # Ensure stereo at final step
    output_path = os.path.join(output_dir, f"{audio_type}_{stem_name}.wav")

    ## original version -> control speech decibel with respect to music
    if audio_type == "speech" and speech_gain is not None:
        gain_db = 20 * np.log10(speech_gain)
        combined_audio = combined_audio.apply_gain(gain_db)


    if audio_type == "noise" and snr is not None and reference_audio_path:
        reference_audio = AudioSegment.from_file(reference_audio_path).set_channels(2)  # Ensure stereo for reference
        calculated_gain = calculate_noise_gain(reference_audio, combined_audio, snr)
        combined_audio = combined_audio.apply_gain(calculated_gain)

    combined_audio.export(output_path, format="wav")
    return output_path

def mix_audios(
    vocal_path: str, background_path: str, speech_path: Optional[str],
    noise_path: Optional[str], output_dir: str, stem_name: str,
    background_gain : float,
) -> str:
    """
    Mix the various audio components into a final mix.

    Args:
        vocal_path (str): Path to the vocal audio.
        background_path (str): Path to the background audio.
        speech_path (Optional[str]): Path to the speech audio.
        noise_path (Optional[str]): Path to the noise audio.
        output_dir (str): Directory to save outputs.
        stem_name (str): Name of the stem file.

    Returns:
        str: Path to the mixed audio file.
    """
    vocals_audio = AudioSegment.from_file(vocal_path)
    backgrounds_audio = AudioSegment.from_file(background_path)

    if background_gain is not None:
        bg_gain_db = 20 * np.log10(background_gain)
        vocals_audio = vocals_audio.apply_gain(bg_gain_db)
        backgrounds_audio = backgrounds_audio.apply_gain(bg_gain_db)

    bg_music_audio = vocals_audio.overlay(backgrounds_audio)

    if speech_path:
        speech_audio = AudioSegment.from_file(speech_path)
        mix_audio = bg_music_audio.overlay(speech_audio)
    else:
        mix_audio = bg_music_audio

    if noise_path:
        noise_audio = AudioSegment.from_file(noise_path)
        mix_audio = mix_audio.overlay(noise_audio)

    mix_path = os.path.join(output_dir, f"mix_{stem_name}.wav")
    mix_audio.export(mix_path, format="wav")

    return mix_path
def create_spleeter_csv(
    stem_files: List[str], stem_dir: str, speech_dir: str, noise_dir: str,
    subset_dir: str, csv_path: str,
    speech_mix_ratio: float = 0.7, speech_gain: float = 1.0,
    noise_mix_ratio: float = 0.5, noise_snr: float = 10,
    background_gain :float = 1.0
) -> None:
    """
    Process a subset of stem files and save outputs into the subset directory.

    Args:
        stem_files (List[str]): List of stem files to process.
        stem_dir (str): Directory containing stem files.
        speech_dir (str): Directory containing speech data.
        noise_dir (str): Directory containing noise data.
        subset_dir (str): Directory to save outputs for this subset.
        csv_path (str): Path to save the subset CSV file.
        speech_mix_ratio (float, optional): Ratio of speech in the mix.
        speech_gain (float, optional): Gain factor for speech audio.
        noise_mix_ratio (float, optional): Ratio of noise in the mix.
        noise_snr (float, optional): Desired SNR for noise.
    """
    data = []
    os.makedirs(subset_dir, exist_ok=True)

    if not stem_files:
        logging.error(f"No stem files provided for {subset_dir}")
        return

    for i, stem_file in enumerate(tqdm(stem_files, desc=f"Processing {subset_dir} stems")):
        stem_path = os.path.join(stem_dir, stem_file)
        stem_name = stem_file.split(".stem")[0]

        try:
            vocal_path, background_path, vocal_duration, rate = extract_and_save_stems(stem_path, subset_dir)

            speech_path = process_audio_data(
                audio_dir=speech_dir,
                vocal_duration=vocal_duration,
                rate=rate,
                stem_name=stem_name,
                mix_ratio=speech_mix_ratio,
                output_dir=subset_dir,
                audio_type='speech',
                speech_gain=speech_gain
            )

            noise_path = process_audio_data(
                audio_dir=noise_dir,
                vocal_duration=vocal_duration,
                rate=rate,
                stem_name=stem_name,
                mix_ratio=noise_mix_ratio,
                output_dir=subset_dir,
                audio_type='noise',
                snr=noise_snr,
                reference_audio_path=background_path
            )
    
            mix_path = mix_audios(
                vocal_path, background_path, speech_path,
                noise_path, subset_dir, stem_name,
                background_gain,
            )

            data.append({
                "mix_path": mix_path,
                "vocals_path": vocal_path,
                "backgrounds_path": background_path,
                "speech_path": speech_path,
                "others_path": noise_path,    # Noise
                "duration": vocal_duration,
            })

            logging.info(f"Processed {stem_file}")

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

    parser = argparse.ArgumentParser(description="Create Spleeter CSV")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--speech_gain', type=float, help='Gain factor for speech audio (default from config)')
    parser.add_argument('--noise_snr', type=float, help='Desired SNR for noise (default from config)')
    parser.add_argument('--background_gain', type=float, help='Gain factor for background audio (default from config)') 
    parser.add_argument('--seed', type=float, help='Set random seed for reproduction')  


    args = parser.parse_args()
    config = load_config(args.config)


    # SEED = 42
    SEED = args.seed if args.seed is not None else config.get('seed', 42)
    random.seed(SEED)

    # Base output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Optional parameters
    speech_gain = args.speech_gain if args.speech_gain is not None else config.get('speech_gain', 1.0)
    noise_snr = args.noise_snr if args.noise_snr is not None else config.get('noise_snr', 10)
    background_gain = args.background_gain if args.background_gain is not None else config.get('background_gain', 1.0)


    # List and shuffle stem files
    stem_files = [f for f in os.listdir(config['stem_dir']) if f.endswith(".stem.mp4")]
    if not stem_files:
        logging.error(f"No stem files found in {config['stem_dir']}")
        return

    random.shuffle(stem_files)

    # Split into train, validation, test
    train_files, val_files, test_files = create_train_val_test_split(stem_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Define directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process train subset
    train_csv_path = os.path.join(output_dir, "train.csv")
    create_spleeter_csv(
        stem_files=train_files,
        stem_dir=config['stem_dir'],
        speech_dir=config['speech_dir'],
        noise_dir=config.get('noise_dir', ''),
        subset_dir=train_dir,
        csv_path=train_csv_path,
        speech_mix_ratio=config.get('speech_mix_ratio', 0.7),
        speech_gain=speech_gain,
        noise_mix_ratio=config.get('noise_mix_ratio', 0.5),
        noise_snr=noise_snr,
        background_gain = background_gain
    )

    # Process validation subset
    validation_csv_path = os.path.join(output_dir, "validation.csv")
    create_spleeter_csv(
        stem_files=val_files,
        stem_dir=config['stem_dir'],
        speech_dir=config['speech_dir'],
        noise_dir=config.get('noise_dir', ''),
        subset_dir=train_dir,  # Validation data also stored in train directory
        csv_path=validation_csv_path,
        speech_mix_ratio=config.get('speech_mix_ratio', 0.7),
        speech_gain=speech_gain,
        noise_mix_ratio=config.get('noise_mix_ratio', 0.5),
        noise_snr=noise_snr
    )

    # Process test subset
    test_csv_path = os.path.join(output_dir, "test.csv")
    create_spleeter_csv(
        stem_files=test_files,
        stem_dir=config['stem_dir'],
        speech_dir=config['speech_dir'],
        noise_dir=config.get('noise_dir', ''),
        subset_dir=test_dir,
        csv_path=test_csv_path,
        speech_mix_ratio=config.get('speech_mix_ratio', 0.7),
        speech_gain=speech_gain,
        noise_mix_ratio=config.get('noise_mix_ratio', 0.5),
        noise_snr=noise_snr,
        background_gain = background_gain
    )

    # Create a total CSV if needed
    total_result_csv_path = os.path.join(output_dir, "total_result.csv")
    all_data = []
    for csv_file in [train_csv_path, validation_csv_path, test_csv_path]:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_data.append(df)
    if all_data:
        total_df = pd.concat(all_data, ignore_index=True)
        total_df.to_csv(total_result_csv_path, index=False)
        logging.info(f"Total CSV file saved to {total_result_csv_path}")
    else:
        logging.warning("No data to write to total_result.csv.")

if __name__ == "__main__":
    main()
