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


def create_train_val_test_split(csv_path: str, result_dir: str) -> None:
    """
    Split the final CSV file into train, validation, and test sets.

    Args:
        csv_path (str): Path to the original CSV file.
        result_dir (str): Directory to save the train/val/test CSV files.
    """
    os.makedirs(result_dir, exist_ok=True)

    # Load the final dataset
    data = pd.read_csv(csv_path)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data

    # Calculate split sizes
    total_size = len(data)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    # Split the data
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    # Save splits to CSV
    train_path = os.path.join(result_dir, "train.csv")
    val_path = os.path.join(result_dir, "validation.csv")
    test_path = os.path.join(result_dir, "test.csv")

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    logging.info(f"Train, validation, and test splits saved to {result_dir}")
    logging.info(f"Train set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")



def calculate_noise_gain(clean_audio: AudioSegment, noise_audio: AudioSegment, snr: float) -> float:
    """
    Calculate the gain needed for the noise audio to achieve the desired SNR.
    This is because we use SNR (constant value) as input.

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

    # Extract stems
    audio, rate = stempeg.read_stems(stem_path)
    vocals = audio[4]
    backgrounds = np.sum(audio[1:4], axis=0)
    mixture = audio[0]

    # 0 - The mixture,
    # 1 - The drums,
    # 2 - The bass,
    # 3 - The rest of the accompaniment,
    # 4 - The vocals.


    # Convert to mono
    mixture_mono = np.mean(mixture, axis=1)
    vocals_mono = np.mean(vocals, axis=1)
    backgrounds_mono = np.mean(backgrounds, axis=1)

    # Trim silence (Because the music doesn't start immediately, if we don't trim, only the voice will be heard at the beginning of the mixed audio.)
    trimmed_indices = librosa.effects.trim(mixture_mono, top_db=20)[1]
    start_idx, end_idx = trimmed_indices
    vocals_trimmed = vocals_mono[start_idx:end_idx]
    backgrounds_trimmed = backgrounds_mono[start_idx:end_idx]

    # Save audio files
    sf.write(vocal_path, vocals_trimmed, rate)
    sf.write(background_path, backgrounds_trimmed, rate)

    vocal_duration = len(vocals_trimmed) / rate

    return vocal_path, background_path, vocal_duration, rate

def process_audio_data(
    audio_dir: str, vocal_duration: float, rate: int, stem_name: str,
    mix_ratio: float, output_dir: str, audio_type: str,
    gain: Optional[float] = None, snr: Optional[float] = None,
    reference_audio_path: Optional[str] = None
) -> Optional[str]:
    """
    Process audio data (speech or noise) and save the combined audio.

    Args:
        audio_dir (str): Directory containing audio files.
        vocal_duration (float): Duration of the vocal audio.
        rate (int): Sample rate.
        stem_name (str): Name of the stem file.
        mix_ratio (float): Ratio of the audio to mix.
        output_dir (str): Directory to save outputs.
        audio_type (str): Type of audio ('speech' or 'noise').
        gain (Optional[float]): Gain to apply to audio (for speech).
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
    random.shuffle(audio_files)

    current_duration = 0
    while current_duration < vocal_duration:
        if total_audio_duration > 0:
            if not audio_files:
                logging.warning(f"Not enough {audio_type} files.")
                break
            audio_file = audio_files.pop()
            audio_segment = AudioSegment.from_file(audio_file)
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
            silence = AudioSegment.silent(duration=silence_length * 1000)
            audio_segments.append(silence)

            current_duration += silence_length
            total_silence_duration -= silence_length

    if current_duration < vocal_duration:
        remaining_silence = vocal_duration - current_duration
        silence = AudioSegment.silent(duration=remaining_silence * 1000)
        audio_segments.append(silence)
        current_duration += remaining_silence

    combined_audio = sum(audio_segments)
    combined_audio = combined_audio.set_frame_rate(rate)
    combined_audio = combined_audio.set_channels(1)
    output_path = os.path.join(output_dir, f"{audio_type}_{stem_name}.wav")

    if audio_type == "speech" and gain is not None:
        # Adjust speech volume
        gain_db = 20 * np.log10(gain)
        combined_audio = combined_audio.apply_gain(gain_db)
    elif audio_type == "noise" and snr is not None and reference_audio_path:
        # Adjust noise volume based on desired SNR
        reference_audio = AudioSegment.from_file(reference_audio_path).set_channels(1)
        calculated_gain = calculate_noise_gain(reference_audio, combined_audio, snr)
        combined_audio = combined_audio.apply_gain(calculated_gain)

    combined_audio.export(output_path, format="wav")

    return output_path

def mix_audios(
    vocal_path: str, background_path: str, speech_path: Optional[str],
    noise_path: Optional[str], output_dir: str, stem_name: str
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
    vocals_audio = AudioSegment.from_file(vocal_path).set_channels(1)
    backgrounds_audio = AudioSegment.from_file(background_path).set_channels(1)

    bg_music_audio = vocals_audio.overlay(backgrounds_audio)

    if speech_path:
        speech_audio = AudioSegment.from_file(speech_path).set_channels(1)
        mix_audio = bg_music_audio.overlay(speech_audio)
    else:
        mix_audio = bg_music_audio

    if noise_path:
        noise_audio = AudioSegment.from_file(noise_path).set_channels(1)
        mix_audio = mix_audio.overlay(noise_audio)

    mix_path = os.path.join(output_dir, f"mix_{stem_name}.wav")
    mix_audio.export(mix_path, format="wav")

    return mix_path

def create_spleeter_csv(
    stem_dir: str, speech_dir: str, noise_dir: str, output_dir: str, csv_path: str,
    speech_mix_ratio: float = 0.7, speech_gain: float = 1.0,
    noise_mix_ratio: float = 0.5, noise_snr: float = 10
) -> None:
    """
    Main function to create the Spleeter CSV file and process audio data.

    Args:
        stem_dir (str): Directory containing stem files.
        speech_dir (str): Directory containing speech data.
        noise_dir (str): Directory containing noise data.
        output_dir (str): Directory to save outputs.
        csv_path (str): Path to save the CSV file.
        speech_mix_ratio (float, optional): Ratio of speech in the mix. Defaults to 0.7.
        speech_gain (float, optional): Gain factor for speech audio. Defaults to 1.0.
        noise_mix_ratio (float, optional): Ratio of noise in the mix. Defaults to 0.5.
        noise_snr (float, optional): Desired SNR for noise. Defaults to 10.
    """
    data = []
    os.makedirs(output_dir, exist_ok=True)

    stem_files = [f for f in os.listdir(stem_dir) if f.endswith(".stem.mp4")]
    if not stem_files:
        logging.error(f"No stem files found in {stem_dir}")
        return


    for i, stem_file in enumerate(tqdm(stem_files, desc="Processing stems")):
        # if i >= 3:
        #     break
        stem_path = os.path.join(stem_dir, stem_file)
        stem_name = stem_file.split(".stem")[0]

        try:
            # Extract vocals and backgrounds
            vocal_path, background_path, vocal_duration, rate = extract_and_save_stems(stem_path, output_dir)

            # Process speech data
            speech_path = process_audio_data(
                audio_dir=speech_dir,
                vocal_duration=vocal_duration,
                rate=rate,
                stem_name=stem_name,
                mix_ratio=speech_mix_ratio,
                output_dir=output_dir,
                audio_type='speech',
                gain=speech_gain
            )

            # Process noise data
            noise_path = process_audio_data(
                audio_dir=noise_dir,
                vocal_duration=vocal_duration,
                rate=rate,
                stem_name=stem_name,
                mix_ratio=noise_mix_ratio,
                output_dir=output_dir,
                audio_type='noise',
                snr=noise_snr,
                reference_audio_path=background_path
            )

            # Mix audios
            mix_path = mix_audios(
                vocal_path, background_path, speech_path,
                noise_path, output_dir, stem_name
            )

            data.append({
                "mix_path": mix_path,
                "vocals_path": vocal_path,
                "backgrounds_path": background_path,
                "speech_path": speech_path,
                "others_path": noise_path,  # Noise
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
        logging.warning("No data to write to CSV.")

def main() -> None:
    """
    Entry point of the script.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Create Spleeter CSV")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--speech_gain', type=float, help='Gain factor for speech audio (default from config)')
    parser.add_argument('--noise_snr', type=float, help='Desired SNR for noise (default from config)')

    args = parser.parse_args()
    config = load_config(args.config)

    # Override values if provided via command line
    speech_gain = args.speech_gain if args.speech_gain is not None else config.get('speech_gain', 1.0)
    noise_snr = args.noise_snr if args.noise_snr is not None else config.get('noise_snr', 10)

    # Create the total result CSV file
    total_result_csv_path = os.path.join(config['result_dir'], 'total_result.csv')
    create_spleeter_csv(
        stem_dir=config['stem_dir'],
        speech_dir=config['speech_dir'],
        noise_dir=config.get('noise_dir', ''),
        output_dir=config['output_dir'],
        csv_path=total_result_csv_path,
        speech_mix_ratio=config.get('speech_mix_ratio', 0.7),
        speech_gain=speech_gain,
        noise_mix_ratio=config.get('noise_mix_ratio', 0.5),
        noise_snr=noise_snr
    )

    # Split final CSV into train, validation, and test sets
    create_train_val_test_split(
        csv_path=total_result_csv_path,
        result_dir=config['result_dir']
    )

if __name__ == "__main__":
    main()