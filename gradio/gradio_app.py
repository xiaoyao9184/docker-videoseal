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
import random
import string
from audioseal import AudioSeal
import videoseal
from videoseal.utils.display import save_video_audio_to_mp4

# Load video_model if not already loaded in reload mode
if 'video_model' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the VideoSeal model
    video_model = videoseal.load("videoseal")
    video_model.eval()
    video_model.to(device)
    video_model_nbytes = int(video_model.embedder.msg_processor.nbits / 8)

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


def load_video(file):
    # Read the video and convert to tensor format
    video, audio, info = torchvision.io.read_video(file, output_format="TCHW", pts_unit="sec")
    assert "audio_fps" in info, "The input video must contain an audio track. Simply refer to the main videoseal inference code if not."

    # Normalize the video frames to the range [0, 1]
    # audio = audio.float()
    # video = video.float() / 255.0

    # Normalize the video frames to the range [0, 1] and trim to 3 second
    fps = 24
    video = video[:fps * 3].float() / 255.0
    
    sample_rate = info["audio_fps"]
    audio = audio[:, :int(sample_rate * 3)].float()

    return video, info["video_fps"], audio, info["audio_fps"]

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

def embed_watermark(output_file, msg_v, msg_a, video_only, video, fps, audio, sample_rate):
    # Perform watermark embedding on video
    with torch.no_grad():
        outputs = video_model.embed(video, is_video=True, msgs=msg_v)

    # Extract the results
    video_w = outputs["imgs_w"]  # Watermarked video frames
    video_msgs = outputs["msgs"]  # Watermark messages

    if not video_only:
        # Resample the audio to 16kHz for watermarking
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)

        # If the audio has more than one channel, average all channels to 1 channel
        if audio_16k.shape[0] > 1:
            audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
        else:
            audio_16k_mono = audio_16k

        # Add batch dimension to the audio tensor
        audio_16k_mono_batched = audio_16k_mono.unsqueeze(0).to(device)

        # Get the watermark for the audio
        with torch.no_grad():
            watermark = audio_generator.get_watermark(
                audio_16k_mono_batched, 16000, message=msg_a
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
    else:
        audio_w = audio

    # for Incompatible pixel format 'rgb24' for codec 'libx264', auto-selecting format 'yuv444p'
    video_w = video_w.flip(1)

    # Save the watermarked video and audio
    save_video_audio_to_mp4(
        video_tensor=video_w,
        audio_tensor=audio_w,
        fps=int(fps),
        audio_sample_rate=int(sample_rate),
        output_filename=output_file,
    )

    print(f"encoded message: \n Audio: {msg_a} \n Video {video_msgs[0]}")

    return video_w, audio_w

def generate_format_string_by_msg_pt(msg_pt, bytes_count):
    hex_length = bytes_count * 2
    binary_int = 0
    for bit in msg_pt:
        binary_int = (binary_int << 1) | int(bit.item())
    hex_string = format(binary_int, f'0{hex_length}x')

    split_hex = [hex_string[i:i + 4] for i in range(0, len(hex_string), 4)]
    format_hex = "-".join(split_hex)

    return hex_string, format_hex

def detect_watermark(video_only, video, audio, sample_rate):
    # Detect watermarks in the video
    with torch.no_grad():
        msg_extracted = video_model.extract_message(video)

    print(f"Extracted message from video: {msg_extracted}")

    if not video_only:
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0).to(device)  # batchify

        # if stereo convert to mono
        if audio.shape[1] > 1:
            audio = torch.mean(audio, dim=1, keepdim=True)

        # Resample the audio to 16kHz for detectting
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)(audio)

        # Detect watermarks in the audio
        with torch.no_grad():
            result, message = audio_detector.detect_watermark(audio_16k, 16000)

            # pred_prob is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
            # A watermarked audio should have pred_prob[:, 1, :] > 0.5
            # message_prob is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
            # message will be a random tensor if the detector detects no watermarking from the audio
            pred_prob, message_prob = audio_detector(audio_16k, sample_rate)

        print(f"Detection result for audio: {result}")
        print(f"Extracted message from audio: {message}")

        return msg_extracted, (result, message, pred_prob, message_prob)
    else:
        return msg_extracted, None

def get_waveform_and_specgram(waveform, sample_rate):
    # If the audio has more than one channel, average all channels to 1 channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"Waveform and specgram")

    return figure

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

with gr.Blocks(title="VideoSeal") as demo:
    gr.Markdown("""
    # VideoSeal Demo

    The current video will be YUV444P encoded, truncated to 3 seconds for use, and multi-channel audio will be merged into a single channel for processing.

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

                            format_like, regex_pattern = generate_hex_format_regex(video_model_nbytes)
                            msg, _ = generate_hex_random_message(video_model_nbytes)
                            embedding_msg = gr.Textbox(
                                label=f"Message ({video_model_nbytes} bytes hex string)",
                                info=f"format like {format_like}",
                                value=msg,
                                interactive=False, show_copy_button=True)
                        with gr.Column():
                            embedding_only_vid = gr.Checkbox(label="Only Video", value=False)

                            embedding_specgram = gr.Checkbox(label="Show specgram", value=False, info="Show debug information")

                            format_like_a, regex_pattern_a = generate_hex_format_regex(audio_generator_nbytes)
                            msg_a, _ = generate_hex_random_message(audio_generator_nbytes)
                            embedding_msg_a = gr.Textbox(
                                label=f"Audio Message ({audio_generator_nbytes} bytes hex string)",
                                info=f"format like {format_like_a}",
                                value=msg_a,
                                interactive=False, show_copy_button=True)

                    embedding_btn = gr.Button("Embed Watermark")
                with gr.Column():
                    marked_vid = gr.Video(label="Output Audio", show_download_button=True)
                    specgram_original = gr.Plot(label="Original Audio", format="png", visible=False)
                    specgram_watermarked = gr.Plot(label="Watermarked Audio", format="png", visible=False)

            def change_embedding_type(video_only):
                return [gr.update(visible=not video_only, value=False),gr.update(visible=not video_only)]
            embedding_only_vid.change(
                fn=change_embedding_type,
                inputs=[embedding_only_vid],
                outputs=[embedding_specgram, embedding_msg_a]
            )

            def change_embedding_type(type):
                if type == "random":
                    msg, _ = generate_hex_random_message(video_model_nbytes)
                    msg_a,_ = generate_hex_random_message(audio_generator_nbytes)
                    return [gr.update(interactive=False, value=msg),gr.update(interactive=False, value=msg_a)]
                else:
                    return [gr.update(interactive=True),gr.update(interactive=True)]
            embedding_type.change(
                fn=change_embedding_type,
                inputs=[embedding_type],
                outputs=[embedding_msg, embedding_msg_a]
            )

            def check_embedding_msg(msg, msg_a):
                if not re.match(regex_pattern, msg):
                    gr.Warning(
                        f"Invalid format. Please use like '{format_like}'",
                        duration=0)
                if not re.match(regex_pattern_a, msg_a):
                    gr.Warning(
                        f"Invalid format. Please use like '{format_like_a}'",
                        duration=0)
            embedding_msg.change(
                fn=check_embedding_msg,
                inputs=[embedding_msg, embedding_msg_a],
                outputs=[]
            )

            def run_embed_watermark(file, video_only, show_specgram, msg, msg_a):
                if file is None:
                    raise gr.Error("No file uploaded", duration=5)
                if not re.match(regex_pattern, msg):
                    raise gr.Error(f"Invalid format. Please use like '{format_like}'", duration=5)
                if not re.match(regex_pattern_a, msg_a):
                    raise gr.Error(f"Invalid format. Please use like '{format_like_a}'", duration=5)

                msg_pt = generate_msg_pt_by_format_string(msg, video_model_nbytes)
                msg_pt_a = generate_msg_pt_by_format_string(msg_a, audio_generator_nbytes)
                video, fps, audio, rate = load_video(file)

                output_path = file + '.marked.mp4'
                _, audio_w = embed_watermark(output_path, msg_pt, msg_pt_a, video_only, video, fps, audio, rate)

                if show_specgram:
                    fig_original = get_waveform_and_specgram(audio, rate)
                    fig_watermarked = get_waveform_and_specgram(audio_w, rate)
                    return [
                        output_path,
                        gr.update(visible=True, value=fig_original),
                        gr.update(visible=True, value=fig_watermarked)]
                else:
                    return [
                        output_path,
                        gr.update(visible=False),
                        gr.update(visible=False)]
            embedding_btn.click(
                fn=run_embed_watermark,
                inputs=[embedding_vid, embedding_only_vid, embedding_specgram, embedding_msg, embedding_msg_a],
                outputs=[marked_vid, specgram_original, specgram_watermarked]
            )

        with gr.TabItem("Detect Watermark"):
            with gr.Row():
                with gr.Column():
                    detecting_vid = gr.Video(label="Input Video")
                    detecting_only_vid = gr.Checkbox(label="Only Video", value=False)
                    detecting_btn = gr.Button("Detect Watermark")
                with gr.Column():
                    predicted_messages = gr.JSON(label="Detected Messages")

            def run_detect_watermark(file, video_only):
                if file is None:
                    raise gr.Error("No file uploaded", duration=5)

                video, _, audio, rate = load_video(file)

                if video_only:
                    msg_extracted, _ = detect_watermark(video_only, video, audio, rate)

                    audio_json = None
                else:
                    msg_extracted, (result, message, pred_prob, message_prob) = detect_watermark(video_only, video, audio, rate)

                    _, fromat_msg = generate_format_string_by_msg_pt(message[0], audio_generator_nbytes)
                    
                    sum_above_05 = (pred_prob[:, 1, :] > 0.5).sum(dim=1)
                    
                    audio_json = {
                        "socre": result,
                        "message": fromat_msg,
                        "frames_count_all": pred_prob.shape[2],
                        "frames_count_above_05": sum_above_05[0].item(),
                        "bits_probability": message_prob[0].tolist(),
                        "bits_massage": message[0].tolist()
                    }

                _, fromat_msg = generate_format_string_by_msg_pt(msg_extracted[0], video_model_nbytes)
                
                # Create message output as JSON
                message_json = {
                    "video": {
                        "message": fromat_msg,
                    },
                    "audio:": audio_json
                }
                return message_json
            detecting_btn.click(
                fn=run_detect_watermark,
                inputs=[detecting_vid, detecting_only_vid],
                outputs=[predicted_messages]
            )

if __name__ == "__main__":
    demo.launch()
