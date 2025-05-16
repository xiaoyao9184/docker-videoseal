import os
import sys

if "APP_PATH" in os.environ:
    app_path = os.path.abspath(os.environ["APP_PATH"])
    if os.getcwd() != app_path:
        # fix sys.path for import
        os.chdir(app_path)
    if app_path not in sys.path:
        sys.path.append(app_path)

import gradio as gr

import torch
import torchaudio
import torchvision
import matplotlib.pyplot as plt
import re
import math
import random
import string
import ffmpeg
import subprocess
import numpy as np
import tqdm
from audioseal import AudioSeal
import videoseal
from videoseal.utils.display import save_video_audio_to_mp4

# Load video_model if not already loaded in reload mode
if 'video_models' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_models = {}

    # Load the VideoSeal model 1.0
    video_model = videoseal.load("videoseal_1.0")
    video_model.eval()
    video_model.to(device)
    video_models['1.0'] = video_model

    # Load the VideoSeal model 0.0
    video_model = videoseal.load("videoseal_0.0")
    video_model.eval()
    video_model.to(device)
    video_models['0.0'] = video_model

# Load the AudioSeal model
# Load audio_generator if not already loaded in reload mode
if 'audio_generator' not in globals():
    audio_generator = AudioSeal.load_generator("audioseal_wm_16bits")
    audio_generator = audio_generator.to(device)
    audio_generator_nbytes = int(audio_generator.msg_processor.nbits / 8)

# Load audio_detector if not already loaded in reload mode
if 'audio_detector' not in globals():
    audio_detector = AudioSeal.load_detector("audioseal_detector_16bits")
    audio_detector = audio_detector.to(device)

def get_model_nbytes(model_version):
    video_model = video_models[model_version]
    return int(video_model.embedder.msg_processor.nbits / 8)

def generate_msg_pt_by_format_string(format_string, bytes_count):
    msg_hex = format_string.replace("-", "")
    hex_length = bytes_count * 2
    binary_list = []
    for i in range(0, len(msg_hex), hex_length):
        chunk = msg_hex[i:i+hex_length]
        binary = bin(int(chunk, 16))[2:].zfill(bytes_count * 8)
        binary_list.append([int(b) for b in binary])
    # torch.randint(0, 2, (1, 16), dtype=torch.int32)
    msg_pt = torch.tensor(binary_list, dtype=torch.int32)
    return msg_pt.to(device)

def generate_format_string_by_msg_pt(msg_pt, bytes_count):
    if msg_pt is None: return '', None
    hex_length = bytes_count * 2
    binary_int = 0
    for bit in msg_pt:
        binary_int = (binary_int << 1) | int(bit.item())
    hex_string = format(binary_int, f'0{hex_length}x')

    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    format_hex = "-".join(split_hex)
    return hex_string, format_hex

def generate_hex_format_regex(bytes_count):
    hex_length = bytes_count * 2
    hex_string = 'F' * hex_length
    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    format_like = "-".join(split_hex)
    regex_pattern = '^' + '-'.join([r'[0-9A-Fa-f]{4}'] * len(split_hex)) + '$'
    return format_like, regex_pattern

def generate_hex_random_message(bytes_count):
    hex_length = bytes_count * 2
    hex_string = ''.join(random.choice(string.hexdigits) for _ in range(hex_length))
    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    random_str = "-".join(split_hex)
    return random_str, "".join(split_hex)

def embed_video_clip(
    model,
    clip: np.ndarray,
    msgs: torch.Tensor
) -> np.ndarray:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).to(device).permute(0, 3, 1, 2) / 255.0
    outputs = model.embed(clip_tensor, msgs=msgs, is_video=True)
    processed_clip = outputs["imgs_w"]
    processed_clip = (processed_clip * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    return processed_clip

def embed_video(
    model,
    input_path: str,
    output_path: str,
    msgs: torch.Tensor,
    chunk_size: int,
    crf: int = 23
) -> None:
    # Read video dimensions
    probe = ffmpeg.probe(input_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    fps = float(video_info['r_frame_rate'].split('/')[0]) / float(video_info['r_frame_rate'].split('/')[1])
    num_frames = int(video_info.get('nb_read_frames', 0))

    # fallback using tags
    if not num_frames and 'DURATION' in video_info.get('tags', {}):
        h, m, s = video_info['tags']['DURATION'].split(':')
        duration = int(h) * 3600 + int(m) * 60 + float(s)
        num_frames = int(duration * fps)

    # Open the input video
    process1 = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=fps)
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )
    # Open the output video
    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p', r=fps, crf=crf)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=subprocess.PIPE)
    )

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frame_count = 0
    pbar = tqdm.tqdm(total=num_frames, unit='frame', desc="Watermark video embedding")
    while True:
        # TODO block EOF on Windows
        in_bytes = process1.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frame_count % chunk_size] = frame
        frame_count += 1
        pbar.update(1)
        if frame_count % chunk_size == 0:
            processed_frame = embed_video_clip(model, chunk, msgs)
            process2.stdin.write(processed_frame.tobytes())

    process1.stdout.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()

    return

def get_sample_size(sample_fmt):
    if sample_fmt == 's16':
        return 2, np.int16
    elif sample_fmt == 's16p':
        return 2, np.float16
    elif sample_fmt == 'flt':
        return 4, np.int32
    elif sample_fmt == 'fltp':
        return 4, np.float32
    elif sample_fmt == 's32':
        return 4, np.int32
    elif sample_fmt == 's32p':
        return 4, np.float32
    elif sample_fmt == 'u8':
        return 1, np.int8
    else:
        raise ValueError(f"Unsupported sample_fmt: {sample_fmt}")

def embed_audio_clip(
    model,
    clip: np.ndarray,
    msgs: torch.Tensor,
    sample_rate
) -> np.ndarray:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).to(device)

    # Resample the audio to 16kHz for watermarking
    audio_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)(clip_tensor)

    # If the audio has more than one channel, average all channels to 1 channel
    if audio_16k.shape[0] > 1:
        audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
    else:
        audio_16k_mono = audio_16k

    # Add batch dimension to the audio tensor
    audio_16k_mono_batched = audio_16k_mono.unsqueeze(0)

    # Get the watermark for the audio
    with torch.no_grad():
        watermark = model.get_watermark(
            audio_16k_mono_batched, 16000, message=msgs
        )

    # Embed the watermark in the audio
    audio_16k_w = audio_16k_mono_batched + watermark

    # Remove batch dimension from the watermarked audio tensor
    audio_16k_w = audio_16k_w.squeeze(0)

    # If the original audio had more than one channel, duplicate the watermarked audio to all channels
    if audio_16k.shape[0] > 1:
        audio_16k_w = audio_16k_w.repeat(audio_16k.shape[0], 1)

    # Resample the watermarked audio back to the original sample rate
    audio_w = torchaudio.transforms.Resample(16000, sample_rate).to(device)(audio_16k_w)

    processed_clip = audio_w.cpu().numpy()
    return processed_clip

def embed_audio(
    model,
    input_path: str,
    output_path: str,
    msgs: torch.Tensor,
    chunk_size: int
) -> None:
    # Read audio dimensions
    probe = ffmpeg.probe(input_path)
    audio_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'audio')
    sample_rate = int(audio_info['sample_rate'])
    sample_fmt = audio_info['sample_fmt']
    channels = int(audio_info['channels'])
    duration = float(audio_info.get('duration', 0))

    # fallback using tags
    if not duration and 'DURATION' in audio_info.get('tags', {}):
        h, m, s = audio_info['tags']['DURATION'].split(':')
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    # CASE 1 Read audio all at once

    # audio_data, stderr_output = (
    #     ffmpeg
    #     .input(input_path, loglevel='debug')
    #     .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sample_rate, ac=channels)
    #     .run(capture_stdout=True, capture_stderr=True)
    # )
    # audio_data = process.stdout.read()
    # print("audio numpy total size:", len(audio_data))
    # process.stdout.close()
    # process.wait()
    # stderr_output = process.stderr.read().decode('utf-8')
    # print(stderr_output)

    # CASE 2 Read async
    # NOTE loglevel='debug' not work on Windows
    # NOTE format='wav' data size(4104768) bigger than format='s16le'(4104688)

    # process = (
    #     ffmpeg
    #     .input(input_path, loglevel='debug')
    #     .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sample_rate, ac=channels)
    #     .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    # )
    # audio_data = process.stdout.read()
    # print("audio numpy total size:", len(audio_data))
    # process.stdout.close()
    # process.wait()
    # stderr_output = process.stderr.read().decode('utf-8')
    # print(stderr_output)

    # stderr_output example:
    #
    # # AVIOContext @ 0x5d878ea02e80] Statistics: 4104688 bytes written, 0 seeks, 251 writeouts
    # # [out#0/f32le @ 0x5d878eaf31c0] Output file #0 (pipe:):
    # # [out#0/f32le @ 0x5d878eaf31c0]   Output stream #0:0 (audio): 251 frames encoded (513086 samples); 251 packets muxed (4104688 bytes);
    # # [out#0/f32le @ 0x5d878eaf31c0]   Total: 251 packets (4104688 bytes) muxed

    # CASE 3 Read by torchaudio
    # NOTE torchvision read audio format is f32le

    # _, audio, info = torchvision.io.read_video(input_path, output_format="TCHW")
    # print("audio numpy total size:", audio.nbytes)


    # Open the input audio
    process1 = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=channels, ar=sample_rate)
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )
    # Open the output audio
    process2 = (
        ffmpeg
        .input('pipe:', format='f32le', ac=channels, ar=sample_rate)
        .output(output_path, format='wav', acodec='pcm_f32le', ac=channels, ar=sample_rate)
        # not work
        # .output(output_path, acodec='libfdk_aac', ac=channels, ar=sample_rate)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=subprocess.PIPE)
    )

    # CASE read all and write all

    # while True:
    #     audio_data = process1.stdout.read()
    #     if not audio_data:
    #         break
    #     try:
    #         process2.stdin.write(audio_data)
    #     except BrokenPipeError:
    #         print("Broken pipe: process2 has closed the input.")
    #         break

    # Process the audio
    sample_size, sample_type = get_sample_size(sample_fmt)
    second_size = sample_size * channels * sample_rate
    chunk = np.zeros((chunk_size, sample_rate, channels), dtype=sample_type)
    second_count = 0
    pbar = tqdm.tqdm(total=math.ceil(duration), unit='second', desc="Watermark audio embedding")
    while True:
        in_bytes = process1.stdout.read(second_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, sample_type)
        frame = frame.reshape((-1, channels))
        chunk[second_count % chunk_size, :len(frame)] = frame
        second_count += 1
        pbar.update(1)
        if second_count % chunk_size == 0:
            if msgs is None:
                process2.stdin.write(in_bytes)
            else:
                clip = np.concatenate(chunk, axis=0).T
                processed_frame = embed_audio_clip(model, clip, msgs, sample_rate)
                process2.stdin.write(processed_frame.T.tobytes())

    process1.stdout.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()

    # CASE print stderr

    # stderr_output1 = process1.stderr.read().decode('utf-8')
    # stderr_output2 = process2.stderr.read().decode('utf-8')
    # print("Process 1 stderr:")
    # print(stderr_output1)
    # print("Process 2 stderr:")
    # print(stderr_output2)
    return

def embed_watermark(input_path, model_version, output_path, msg_v, msg_a, video_only, progress):
    output_path_video = output_path + ".video.mp4"
    video_model = video_models[model_version]
    embed_video(video_model, input_path, output_path_video, msg_v, 16)

    output_path_audio = output_path + ".audio.m4a"
    if video_only:
        msg_a = None
    embed_audio(audio_generator, input_path, output_path_audio, msg_a, 3)

    # Use FFmpeg to add audio to the video
    final_command = [
        'ffmpeg',
        '-i', output_path_video,
        '-i', output_path_audio,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y', output_path
    ]
    subprocess.run(final_command, check=True)
    return

def detect_video_clip(
    model,
    clip: np.ndarray
) -> torch.Tensor:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    outputs = model.detect(clip_tensor, is_video=True)
    output_bits = outputs["preds"][:, 1:]  # exclude the first which may be used for detection
    return output_bits

def detect_video(
    model,
    version: str,
    input_path: str,
    chunk_size: int
) -> None:
    # Read video dimensions
    probe = ffmpeg.probe(input_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    fps = float(video_info['r_frame_rate'].split('/')[0]) / float(video_info['r_frame_rate'].split('/')[1])
    num_frames = int(video_info['nb_frames'])

    # Open the input video
    process1 = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=fps)
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frame_count = 0
    soft_msgs = []
    pbar = tqdm.tqdm(total=num_frames, unit='frame', desc=f"{version}: Watermark video detecting")
    while True:
        in_bytes = process1.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frame_count % chunk_size] = frame
        frame_count += 1
        pbar.update(1)
        if frame_count % chunk_size == 0:
            soft_msgs.append(detect_video_clip(model, chunk))

    process1.stdout.close()
    process1.wait()

    soft_msgs = torch.cat(soft_msgs, dim=0)
    return soft_msgs

def detect_audio_clip(
    model,
    clip: np.ndarray,
    sample_rate
) -> torch.Tensor:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).to(device)

    # Resample the audio to 16kHz for watermarking
    audio_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)(clip_tensor)

    # If the audio has more than one channel, average all channels to 1 channel
    if audio_16k.shape[0] > 1:
        audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
    else:
        audio_16k_mono = audio_16k

    # Add batch dimension to the audio tensor
    audio_16k_mono_batched = audio_16k_mono.unsqueeze(0)

    # Detect watermarks in the audio
    with torch.no_grad():
        result, message = model.detect_watermark(
            audio_16k_mono_batched, 16000
        )

        # pred_prob is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
        # A watermarked audio should have pred_prob[:, 1, :] > 0.5
        # message_prob is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
        # message will be a random tensor if the detector detects no watermarking from the audio
        pred_prob, message_prob = model(audio_16k_mono_batched, sample_rate)

    # print(f"Detection result for audio: {result}")
    # _, format_msg = generate_format_string_by_msg_pt(message[0], audio_generator_nbytes)
    # print(f"Extracted message from audio: {message}: {format_msg}")
    # print(f"Extracted pred_prob from audio: {pred_prob.shape}")
    # print(f"Extracted message_prob from audio: {message_prob}")
    # print(f"Extracted shape from audio 16k: {audio_16k_mono_batched.shape}")
    # print(f"Extracted shape from audio original: {clip_tensor.shape}")
    return result, message, pred_prob, message_prob

def detect_audio(
    model,
    input_path: str,
    chunk_size: int
) -> None:
    # Read audio dimensions
    probe = ffmpeg.probe(input_path)
    audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
    if len(audio_streams) == 0:
        gr.Warning("No audio stream found in the input file.")
        return None, None, None, None
    audio_info = audio_streams[0]
    sample_rate = int(audio_info['sample_rate'])
    sample_fmt = audio_info['sample_fmt']
    channels = int(audio_info['channels'])
    duration = float(audio_info['duration'])

    # Open the input audio
    process1 = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=channels, ar=sample_rate)
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )

    # Process the audio
    sample_size, sample_type = get_sample_size(sample_fmt)
    second_size = sample_size * channels * sample_rate
    chunk = np.zeros((chunk_size, sample_rate, channels), dtype=sample_type)
    second_count = 0
    soft_result = []
    soft_message = []
    soft_pred_prob = []
    soft_message_prob = []
    pbar = tqdm.tqdm(total=math.ceil(duration), unit='second', desc="Watermark audio detecting")
    while True:
        in_bytes = process1.stdout.read(second_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, sample_type)
        frame = frame.reshape((-1, channels))
        chunk[second_count % chunk_size, :len(frame)] = frame
        second_count += 1
        pbar.update(1)
        if second_count % chunk_size == 0:
            clip = np.concatenate(chunk, axis=0).T
            # print(f"Detection audio second: {second_count-chunk_size}-{second_count}")
            result, message, pred_prob, message_prob = detect_audio_clip(model, clip, sample_rate)
            soft_result.append(result)
            soft_message.append(message)
            soft_pred_prob.append(pred_prob)
            soft_message_prob.append(message_prob)

    process1.stdout.close()
    process1.wait()

    soft_message = torch.cat(soft_message, dim=0)
    soft_pred_prob = torch.cat(soft_pred_prob, dim=0)
    soft_message_prob = torch.cat(soft_message_prob, dim=0)
    return (soft_result, soft_message, soft_pred_prob, soft_message_prob)

def detect_watermark(input_path, version_keys, video_only):
    msgs_v_most = {}
    msgs_v_avg = {}
    msgs_v_frame = {}
    for video_version, video_model in video_models.items():
        if video_version not in version_keys:
            continue
        version_msgs_v_frame = detect_video(video_model, video_version, input_path, 16)
        version_msgs_v_frame = (version_msgs_v_frame > 0).to(int)
        version_msgs_v_avg = (version_msgs_v_frame.to(torch.float32).mean(dim=0) > 0).to(int)
        version_msgs_v_most = None
        version_msgs_v_unique, version_msgs_v_counts = torch.unique(version_msgs_v_frame, dim=0, return_counts=True)
        if len(version_msgs_v_frame) > len(version_msgs_v_counts) > 0:
            version_msgs_v_most_idx = torch.argmax(version_msgs_v_counts)
            version_msgs_v_most = version_msgs_v_unique[version_msgs_v_most_idx]    

        msgs_v_most[video_version] = version_msgs_v_most
        msgs_v_avg[video_version] = version_msgs_v_avg
        msgs_v_frame[video_version] = version_msgs_v_frame

    msgs_a_most = msgs_a_res = msgs_a_frame = msgs_a_pred = msgs_a_prob = None
    if not video_only:
        msgs_a_res, msgs_a_frame, msgs_a_pred, msgs_a_prob = detect_audio(audio_detector, input_path, 1)
        if msgs_a_res is not None:
            msgs_a_res_not_zero = [i for i, x in enumerate(msgs_a_res) if x > 0.5]
            msgs_a_frame_not_zero = msgs_a_frame[msgs_a_res_not_zero]
            msgs_a_unique, msgs_a_counts = torch.unique(msgs_a_frame_not_zero, dim=0, return_counts=True)
            if len(msgs_a_counts) > 0:
                msgs_a_most_idx = torch.argmax(msgs_a_counts)
                msgs_a_most = msgs_a_unique[msgs_a_most_idx]

    return msgs_v_most, msgs_v_avg, msgs_v_frame, msgs_a_most, msgs_a_res, msgs_a_frame, msgs_a_pred, msgs_a_prob


with gr.Blocks(title="VideoSeal") as demo:
    gr.Markdown("""
    # VideoSeal Demo
    ![](https://badge.mcpx.dev?type=server 'MCP Server')
    For video, each frame will be watermarked and detected.
    For audio, each 3 seconds will be watermarked, and each second will be detected.

    **NOTE: The watermarked process will modify both audio and video.
    The video will be re-encoded to yuv420p using libx264,
    and the audio will be duplicated from mono 16kHz and resampled back to the original channel sample rate.**

    Find the project [here](https://github.com/facebookresearch/videoseal.git).
    """)

    with gr.Tabs():
        with gr.TabItem("Embed Watermark"):
            with gr.Row():
                with gr.Column():
                    embedding_vid = gr.Video(label="Input Video")

                    with gr.Row():
                        with gr.Column():
                            embedding_type = gr.Radio(["random", "input"], value="random", label="Type", info="Type of watermarks")

                            video_model_nbytes = get_model_nbytes(list(video_models.keys())[0])
                            format_like_v, _ = generate_hex_format_regex(video_model_nbytes)
                            msg_v, _ = generate_hex_random_message(video_model_nbytes)
                            embedding_msg_v = gr.Textbox(
                                label=f"Message ({video_model_nbytes} bytes hex string)",
                                info=f"format like {format_like_v}",
                                value=msg_v,
                                interactive=False, show_copy_button=True)
                        with gr.Column():
                            embedding_version = gr.Dropdown(video_models.keys(), label="Model version", interactive=True)
                            with gr.Column():
                                embedding_only_vid = gr.Checkbox(label="Only Video", value=False)

                                format_like_a, _ = generate_hex_format_regex(audio_generator_nbytes)
                                msg_a, _ = generate_hex_random_message(audio_generator_nbytes)
                                embedding_msg_a = gr.Textbox(
                                    label=f"Audio Message ({audio_generator_nbytes} bytes hex string)",
                                    info=f"format like {format_like_a}",
                                    value=msg_a,
                                    interactive=False, show_copy_button=True)
                    embedding_btn = gr.Button("Embed Watermark")
                with gr.Column():
                    marked_vid = gr.Video(label="Output Audio", show_download_button=True)

            def change_embedding_silent(video_only):
                return gr.update(visible=not video_only)
            embedding_only_vid.change(
                fn=change_embedding_silent,
                inputs=[embedding_only_vid],
                outputs=[embedding_msg_a],
                api_name=False
            )

            def change_embedding_version(version):
                video_model_nbytes = get_model_nbytes(version)
                format_like_v, _ = generate_hex_format_regex(video_model_nbytes)
                msg_v, _ = generate_hex_random_message(video_model_nbytes)
                return gr.update(
                    label=f"Message ({video_model_nbytes} bytes hex string)",
                    info=f"format like {format_like_v}",
                    value=msg_v)
            embedding_version.change(
                fn=change_embedding_version,
                inputs=[embedding_version],
                outputs=[embedding_msg_v],
                api_name=False
            )

            def change_embedding_type(type, version):
                if type == "random":
                    video_model_nbytes = get_model_nbytes(version)
                    msg_v, _ = generate_hex_random_message(video_model_nbytes)
                    msg_a, _ = generate_hex_random_message(audio_generator_nbytes)
                    return [gr.update(interactive=False, value=msg_v),gr.update(interactive=False, value=msg_a)]
                else:
                    return [gr.update(interactive=True),gr.update(interactive=True)]
            embedding_type.change(
                fn=change_embedding_type,
                inputs=[embedding_type, embedding_version],
                outputs=[embedding_msg_v, embedding_msg_a],
                api_name=False
            )

            def check_embedding_msg(version_v, msg_v, msg_a):
                video_model_nbytes = get_model_nbytes(version_v)
                _, regex_pattern_v = generate_hex_format_regex(video_model_nbytes)
                _, regex_pattern_a = generate_hex_format_regex(audio_generator_nbytes)
                if not re.match(regex_pattern_v, msg_v):
                    gr.Warning(
                        f"Invalid format. Please use like '{format_like_v}'",
                        duration=0)
                if not re.match(regex_pattern_a, msg_a):
                    gr.Warning(
                        f"Invalid format. Please use like '{format_like_a}'",
                        duration=0)
            embedding_msg_v.change(
                fn=check_embedding_msg,
                inputs=[embedding_version, embedding_msg_v, embedding_msg_a],
                outputs=[],
                api_name=False
            )
            embedding_msg_a.change(
                fn=check_embedding_msg,
                inputs=[embedding_msg_v, embedding_msg_a],
                outputs=[],
                api_name=False
            )

            def run_embed_watermark(file, model_version, video_only, msg_v, msg_a, progress=gr.Progress(track_tqdm=True)):
                """
                Embeds a watermark into the given video file using the specified model.

                Args:
                    file (str): Path to the input video file.
                    model_version (str): Identifier for the video model version or checkpoint used for embedding.
                    video_only (bool): If True, embeds watermark only in the video stream; audio is ignored.
                    msg_v (str): A 12- or 32-byte hexadecimal string to embed as a watermark in the video stream (e.g., "FFFF").
                    msg_a (str): A 2-byte hexadecimal string to embed as a watermark in the audio stream (e.g., "FFFF").
                    progress (gr.Progress, optional): Gradio progress tracker for monitoring embedding progress. Defaults to tracking tqdm.

                Returns:
                    str: File path to the watermarked output video file.
                """
                video_model_nbytes = get_model_nbytes(model_version)
                _, regex_pattern_v = generate_hex_format_regex(video_model_nbytes)
                _, regex_pattern_a = generate_hex_format_regex(audio_generator_nbytes)
                if file is None:
                    raise gr.Error("No file uploaded", duration=5)
                if not re.match(regex_pattern_v, msg_v):
                    raise gr.Error(f"Invalid format. Please use like '{format_like_v}'", duration=5)
                if not re.match(regex_pattern_a, msg_a):
                    raise gr.Error(f"Invalid format. Please use like '{format_like_a}'", duration=5)

                msg_pt_v = generate_msg_pt_by_format_string(msg_v, video_model_nbytes)
                msg_pt_a = generate_msg_pt_by_format_string(msg_a, audio_generator_nbytes)

                if video_only:
                    output_path = os.path.join(os.path.dirname(file), "__".join([msg_v]) + '.mp4')
                else:
                    output_path = os.path.join(os.path.dirname(file), "__".join([msg_v, msg_a]) + '.mp4')
                embed_watermark(file, model_version, output_path, msg_pt_v, msg_pt_a, video_only, progress)

                return output_path
            embedding_btn.click(
                fn=run_embed_watermark,
                inputs=[embedding_vid, embedding_version, embedding_only_vid, embedding_msg_v, embedding_msg_a],
                outputs=[marked_vid]
            )

        with gr.TabItem("Detect Watermark"):
            with gr.Row():
                with gr.Column():
                    detecting_vid = gr.Video(label="Input Video")
                    with gr.Row():
                        detecting_model_dd = gr.Dropdown(video_models.keys(), value=list(video_models.keys()), multiselect=True, label="Model version", interactive=True)
                        detecting_only_vid = gr.Checkbox(label="Only Video", value=False)
                    detecting_btn = gr.Button("Detect Watermark")
                with gr.Column():
                    predicted_messages = gr.JSON(label="Detected Messages")

            def run_detect_watermark(file, model_versions, video_only, progress=gr.Progress(track_tqdm=True)):
                """
                Detects a watermark in the given video file using specified model versions.

                Args:
                    file (str): Path to the input video file.
                    model_versions (List[str]): List of model version identifiers (e.g., checkpoint versions) to use for detection.
                    video_only (bool): If True, only the video stream is considered; audio is ignored.
                    progress (gr.Progress, optional): Gradio Progress tracker for visualizing progress. Defaults to tracking tqdm.

                Returns:
                    str: A Markdown-formatted string containing the detection results.
                """
                if file is None:
                    raise gr.Error("No file uploaded", duration=5)

                msgs_v_most, msgs_v_avg, msgs_v_frame, msgs_a_most, msgs_a_res, msgs_a_frame, msgs_a_pred, msgs_a_prob = detect_watermark(file, model_versions, video_only)

                video_json = {}
                for (version_name, version_msgs_v_most), (_, version_msgs_v_avg), (_, version_msgs_v_frame) in zip(msgs_v_most.items(), msgs_v_avg.items(), msgs_v_frame.items()):
                    if version_name not in model_versions:
                        continue

                    video_model_nbytes = get_model_nbytes(version_name)
                    _, format_msg_v_most = generate_format_string_by_msg_pt(version_msgs_v_most, video_model_nbytes)
                    _, format_msg_v_avg = generate_format_string_by_msg_pt(version_msgs_v_avg, video_model_nbytes)
                    format_msg_v_frames = {}
                    for idx, msg in enumerate(version_msgs_v_frame):
                        _, format_msg = generate_format_string_by_msg_pt(msg, video_model_nbytes)
                        format_msg_v_frames[f"{idx}"] = format_msg
                    video_json[version_name] = {
                        "most": format_msg_v_most,
                        "avg": format_msg_v_avg,
                        "frames": format_msg_v_frames
                    }

                if msgs_a_res is None:
                    audio_json = None
                else:
                    _, format_msg_a_most = generate_format_string_by_msg_pt(msgs_a_most, audio_generator_nbytes)
                    format_msg_a_seconds = {}
                    for idx, (result, message, pred_prob, message_prob) in enumerate(zip(msgs_a_res, msgs_a_frame, msgs_a_pred, msgs_a_prob)):
                        _, format_msg = generate_format_string_by_msg_pt(message, audio_generator_nbytes)

                        sum_above_05 = (pred_prob[1, :] > 0.5).sum(dim=0)
                        format_msg_a_seconds[f"{idx}"] = {
                            "socre": result,
                            "message": format_msg,
                            "frames_count_all": pred_prob.shape[1],
                            "frames_count_above_05": sum_above_05.item(),
                            "bits_probability": message_prob.tolist(),
                            "bits_massage": message.tolist()
                        }
                    audio_json = {
                        "most": format_msg_a_most,
                        "seconds": format_msg_a_seconds
                    }

                # Create message output as JSON
                message_json = {
                    "video": video_json,
                    "audio:": audio_json
                }
                return message_json
            detecting_btn.click(
                fn=run_detect_watermark,
                inputs=[detecting_vid, detecting_model_dd, detecting_only_vid],
                outputs=[predicted_messages]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, mcp_server=True, ssr_mode=False)
