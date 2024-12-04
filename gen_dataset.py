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

def calculate_noise_gain(music_audio: AudioSegment, speech_audio: Optional[AudioSegment], noise_audio: AudioSegment, snr: float) -> float:
    """
    Calculate the gain needed for the noise audio to achieve the desired SNR.

    Args:
        music_audio (AudioSegment): The music audio segment.
        speech_audio (Optional[AudioSegment]): The speech audio segment (optional if only music is used).
        noise_audio (AudioSegment): The noise audio segment.
        snr (float): The desired signal-to-noise ratio in dB.

    Returns:
        float: The gain in dB to apply to the noise audio.
    """
    # Calculate RMS for music + speech combined
    music_rms = music_audio.rms
    speech_rms = speech_audio.rms if speech_audio else 0
    combined_rms = np.sqrt((music_rms ** 2 + speech_rms ** 2) / 2)

    # Calculate RMS for noise
    noise_rms = noise_audio.rms

    if noise_rms == 0:
        logging.warning("Noise RMS is zero. Returning infinite gain.")
        return -float('inf')  # Handle silence in noise

    # Desired noise RMS for the given SNR
    desired_noise_rms = combined_rms / (10 ** (snr / 20))

    # Calculate gain
    gain = 20 * np.log10(desired_noise_rms / noise_rms)
    return gain


    # SNR = 20 dB: The signal power is 100 times greater than the noise power.
    # SNR = 0 dB: The signal and noise have equal power.
    # SNR < 0 dB: The noise power is greater than the signal power.



def calculate_speech_gain(music_audio: AudioSegment, speech_audio: AudioSegment, smr: float) -> float:
    """
    Calculate the gain needed for the speech audio to achieve the desired SMR.

    Args:
        music_audio (AudioSegment): The music audio segment.
        speech_audio (AudioSegment): The speech audio segment.
        smr (float): The desired Speech-to-Music Ratio (in dB).

    Returns:
        float: The gain in dB to apply to the speech audio.
    """
    music_rms = music_audio.rms
    speech_rms = speech_audio.rms
    if music_rms == 0 or speech_rms == 0:
        logging.warning("RMS value is zero for either music or speech. Gain cannot be calculated.")
        return 0.0

    # Calculate the desired speech RMS based on SMR
    desired_speech_rms = music_rms * (10 ** (smr / 20))
    gain = 20 * np.log10(desired_speech_rms / speech_rms)

    return gain


def extract_music_stem(stem_path: str, output_dir: str) -> tuple:
    """
    Extract music stem from a stem file and save it.

    Args:
        stem_path (str): Path to the stem file.
        output_dir (str): Directory to save the outputs.

    Returns:
        tuple: Path to the music audio file, duration, and sample rate.
    """
    stem_name = os.path.basename(stem_path).split(".stem")[0]
    music_path = os.path.join(output_dir, f"music_{stem_name}.wav")

    # Extract music stem
    audio, rate = stempeg.read_stems(stem_path)
    # music = np.sum(audio[1:4], axis=0)  # Combine drums, bass, and accompaniment
    music =audio[0]

    # Trim silence
    mixture_mono = np.mean(audio[0], axis=1)
    trimmed_indices = librosa.effects.trim(mixture_mono, top_db=20)[1]
    start_idx, end_idx = trimmed_indices
    music_trimmed = music[start_idx:end_idx, :]

    # Save music file
    sf.write(music_path, music_trimmed, rate)
    music_duration = (end_idx - start_idx) / rate

    return music_path, music_duration, rate

    # 0 - The mixture,
    # 1 - The drums,
    # 2 - The bass,
    # 3 - The rest of the accompaniment,
    # 4 - The vocals.
def process_audio_data(
    audio_dir: str, transcriptions: dict, vocal_duration: float, rate: int, stem_name: str,
    output_dir: str, audio_type: str, min_snr: Optional[float] = None, max_snr: Optional[float] = None,
    min_smr: Optional[float] = None, max_smr: Optional[float] = None, 
    music_audio: Optional[AudioSegment] = None, speech_audio: Optional[AudioSegment] = None
) -> tuple:
    """
    Process audio data (speech or noise) and save the combined audio as stereo.

    Args:
        audio_dir (str): Directory containing audio files.
        transcriptions (dict): Dictionary mapping audio IDs to transcriptions.
        vocal_duration (float): Duration of the vocal audio.
        rate (int): Sample rate.
        stem_name (str): Name of the stem file.
        output_dir (str): Directory to save outputs.
        audio_type (str): Type of audio ('speech' or 'noise').
        min_snr (Optional[float]): Minimum SNR value for noise adjustment.
        max_snr (Optional[float]): Maximum SNR value for noise adjustment.
        min_smr (Optional[float]): Minimum SMR value for speech adjustment.
        max_smr (Optional[float]): Maximum SMR value for speech adjustment.
        music_audio (Optional[AudioSegment]): Music audio for reference (needed for noise SNR adjustment).
        speech_audio (Optional[AudioSegment]): Speech audio for reference (needed for noise SNR adjustment).

    Returns:
        tuple: (Path to the processed audio file, Combined transcription)
    """
    if not audio_dir:
        return None, ""

    audio_segments = []
    combined_transcription = ""

    file_extension = ".flac" if audio_type == "speech" else ".wav"
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(audio_dir)
        for file in files if file.endswith(file_extension)
    ]
    if not audio_files:
        logging.error(f"No {audio_type} files found in {audio_dir}")
        return None, ""

    random.shuffle(audio_files)
    current_duration = 0

    while current_duration < vocal_duration:
        if not audio_files:
            logging.warning(f"Not enough {audio_type} files. Remaining duration will be silent.")
            break

        # Select an audio file
        audio_file = audio_files.pop(0)
        audio_segment = AudioSegment.from_file(audio_file)
        audio_segment = audio_segment.set_channels(2)  # Ensure stereo

        # Skip files longer than 30 seconds
        if audio_segment.duration_seconds > 30:
            logging.info(f"Skipping file {audio_file} as it is longer than 30 seconds.")
            continue

        # Check if adding this audio exceeds the allowed limit
        if current_duration + audio_segment.duration_seconds > vocal_duration:
            break

        # Apply SNR for noise
        if audio_type == "noise" and music_audio is not None and min_snr is not None and max_snr is not None:
            snr = random.uniform(min_snr, max_snr)
            calculated_gain = calculate_noise_gain(music_audio, speech_audio, audio_segment, snr)
            audio_segment = audio_segment.apply_gain(calculated_gain)

        # Apply SMR for speech
        if audio_type == "speech" and min_smr is not None and max_smr is not None:
            smr = random.uniform(min_smr, max_smr)
            if music_audio:
                calculated_gain = calculate_speech_gain(music_audio, audio_segment, smr)
                audio_segment = audio_segment.apply_gain(calculated_gain)
            # Add transcription for speech
            audio_id = os.path.splitext(os.path.basename(audio_file))[0]
            combined_transcription += transcriptions.get(audio_id, "") + " "

        # Add the audio segment
        audio_segments.append(audio_segment)
        current_duration += audio_segment.duration_seconds

        # Add random silence between segments
        silence_length = random.uniform(3, 5)
        if current_duration + silence_length < vocal_duration:
            silence = AudioSegment.silent(duration=silence_length * 1000).set_channels(2)
            audio_segments.append(silence)
            current_duration += silence_length

    # Combine all audio segments
    combined_audio = sum(audio_segments)
    combined_audio = combined_audio.set_frame_rate(rate)
    combined_audio = combined_audio.set_channels(2)  # Ensure stereo at final step

    output_path = os.path.join(output_dir, f"{audio_type}_{stem_name}.wav")
    combined_audio.export(output_path, format="wav")

    return output_path, combined_transcription.strip()


def mix_audios(
    music_path: str, speech_path: Optional[str], noise_path: Optional[str],
    output_dir: str, stem_name: str, smr: float
) -> str:
    """
    Mix the speech, music, and noise components into a final mix.

    Args:
        music_path (str): Path to the music audio.
        speech_path (Optional[str]): Path to the speech audio.
        noise_path (Optional[str]): Path to the noise audio.
        output_dir (str): Directory to save outputs.
        stem_name (str): Name of the stem file.
        smr (float): Speech-to-Music Ratio (SMR) in dB.

    Returns:
        str: Path to the mixed audio file.
    """
    # Load the music audio
    music_audio = AudioSegment.from_file(music_path)

    # Start with music as the base
    mix_audio = music_audio

    # Overlay speech with adjusted gain if available
    if speech_path:
        speech_audio = AudioSegment.from_file(speech_path)
        speech_gain = calculate_speech_gain(music_audio, speech_audio, smr)  # Calculate gain for SMR
        speech_audio = speech_audio.apply_gain(speech_gain)
        mix_audio = mix_audio.overlay(speech_audio)  # Overlay speech on music

    # Overlay noise if available
    if noise_path:
        noise_audio = AudioSegment.from_file(noise_path)
        mix_audio = mix_audio.overlay(noise_audio)  # Overlay noise on top

    # Save the final mixed audio
    mix_path = os.path.join(output_dir, f"mix_{stem_name}.wav")
    mix_audio.export(mix_path, format="wav")

    logging.info(f"Mixed audio saved at {mix_path}")
    return mix_path





def load_transcriptions(trans_dir: str) -> dict:
    """
    Load transcriptions from .trans.txt files.

    Args:
        trans_dir (str): Directory containing transcription files.

    Returns:
        dict: A dictionary mapping audio file names to transcriptions.
    """
    transcriptions = {}
    for root, _, files in os.walk(trans_dir):
        for file in files:
            if file.endswith(".trans.txt"):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_id, transcription = parts
                            transcriptions[audio_id] = transcription


    return transcriptions
def create_spleeter_csv(
    stem_files: List[str], stem_dir: str, speech_dir: str, noise_dir: str,
    subset_dir: str, csv_path: str, trans_dir: str,
    min_smr: float = 20.0, max_smr: float = 30.0,
    min_snr: float = 20.0, max_snr: float = 30.0
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
        trans_dir (str): Directory containing transcription files.
        min_smr (float, optional): Minimum Speech-to-Music Ratio (SMR) in dB.
        max_smr (float, optional): Maximum Speech-to-Music Ratio (SMR) in dB.
        min_snr (float, optional): Minimum Signal-to-Noise Ratio (SNR) in dB.
        max_snr (float, optional): Maximum Signal-to-Noise Ratio (SNR) in dB.
    """
    # Load transcriptions
    transcriptions = load_transcriptions(trans_dir)
    data = []
    os.makedirs(subset_dir, exist_ok=True)

    if not stem_files:
        logging.error(f"No stem files provided for {subset_dir}")
        return

    for i, stem_file in enumerate(tqdm(stem_files, desc=f"Processing {subset_dir} stems")):
        stem_path = os.path.join(stem_dir, stem_file)
        stem_name = stem_file.split(".stem")[0]

        try:
            # Extract music stem
            music_path, music_duration, rate = extract_music_stem(stem_path, subset_dir)
            music_audio = AudioSegment.from_file(music_path)

            # Process speech
            speech_path, combined_transcription = process_audio_data(
                audio_dir=speech_dir,
                transcriptions=transcriptions,
                vocal_duration=music_duration,
                rate=rate,
                stem_name=stem_name,
                output_dir=subset_dir,
                audio_type='speech',
                min_smr=min_smr,
                max_smr=max_smr,
                music_audio=music_audio  # Pass music audio for SMR
            )
            speech_audio = AudioSegment.from_file(speech_path) if speech_path else None

            # Process others (noise)
            others_path, _ = process_audio_data(
                audio_dir=noise_dir,
                transcriptions={},
                vocal_duration=music_duration,
                rate=rate,
                stem_name=stem_name,
                output_dir=subset_dir,
                audio_type='noise',
                min_snr=min_snr,
                max_snr=max_snr,
                music_audio=music_audio,  # Pass music audio for SNR
                speech_audio=speech_audio  # Pass speech audio for combined SNR
            )

            # Create mix audio with SMR
            mix_path = mix_audios(
                music_path=music_path,
                speech_path=speech_path,
                noise_path=others_path,
                output_dir=subset_dir,
                stem_name=stem_name,
                smr=random.uniform(min_smr, max_smr)  # Random SMR for the mix
            )

            # Append data to CSV
            data.append({
                "speech_path": speech_path,
                "music_path": music_path,
                "others_path": others_path,
                "mix_path": mix_path,
                "duration": music_duration,
                "transcription": combined_transcription,
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
    parser.add_argument('--smr', type=float, help='Speech-to-Music Ratio (SMR) in dB (default from config)')
    parser.add_argument('--snr', type=float, help='Desired SNR for noise (default from config)')
    parser.add_argument('--seed', type=float, help='Set random seed for reproduction')

    args = parser.parse_args()
    config = load_config(args.config)


    # SEED = 42
    SEED = args.seed if args.seed is not None else config.get('seed', 42)
    random.seed(SEED)

    # Base output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)



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
        trans_dir=config['trans_dir'],
        min_smr=config.get('min_smr', 20),
        max_smr=config.get('max_smr', 30),
        min_snr=config.get('min_snr', 20),
        max_snr=config.get('max_snr', 30)
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
        trans_dir=config['trans_dir'],
        min_smr=config.get('min_smr', 20),
        max_smr=config.get('max_smr', 30),
        min_snr=config.get('min_snr', 20),
        max_snr=config.get('max_snr', 30)
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
        trans_dir=config['trans_dir'],
        min_smr=config.get('min_smr', 20),
        max_smr=config.get('max_smr', 30),
        min_snr=config.get('min_snr', 20),
        max_snr=config.get('max_snr', 30)
    )
if __name__ == "__main__":
    main()
