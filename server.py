import asyncio
import json
import threading

import numpy as np
import websockets
from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import torch
from TTS.api import TTS
from groq import Groq
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import os
import base64
import re
import uuid


def strip_markdown_and_special_chars(text):
    # First, strip markdown links and formatting
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # Removes markdown links
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # Removes inline code blocks
    text = re.sub(r"\*\*(.*?)\*\*|\*(.*?)\*", r"\1\2", text)  # Removes bold and italics
    text = re.sub(r"#+ ", "", text)  # Removes headers

    #  remove any characters that are not alphanumeric, whitespace, or punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"(){}[\]:;-]', "", text)

    return text


def GetUUID():
    return str(uuid.uuid4())


if __name__ == "__main__":

    print("Starting server, please wait...")

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    # print(TTS().list_models())

    # Init TTS
    tts = TTS("tts_models/en/vctk/vits").to(device)

    recorder = None
    recorder_ready = threading.Event()
    client_websocket = None
    current_uuid = GetUUID()

    history: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """
                You are a helpful chatbot.
                Keep your responses as short and concise as possible.
                Your responses will be used in text to speech pipeline,
                so respond in plain text with punctuations and no extra formatting.
                Respond with a single sentence when possible.""",
        },
    ]

    groq_client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    async def send_to_client(message):
        if client_websocket:
            await client_websocket.send(message)

    def text_detected(text):
        asyncio.new_event_loop().run_until_complete(
            send_to_client(json.dumps({"type": "realtime", "text": text}))
        )
        print(f"\r{text}", flush=True, end="")

    recorder_config = {
        "spinner": False,
        "use_microphone": False,
        "model": "large-v3",
        "language": "en",
        "silero_sensitivity": 0.4,
        "webrtc_sensitivity": 2,
        "post_speech_silence_duration": 0.5,
        "min_length_of_recording": 0,
        "min_gap_between_recordings": 0,
        "enable_realtime_transcription": True,
        "realtime_processing_pause": 0,
        "realtime_model_type": "tiny.en",
        "on_realtime_transcription_stabilized": text_detected,
    }

    def recorder_thread():
        global recorder
        print("Initializing RealtimeSTT...")
        recorder = AudioToTextRecorder(**recorder_config)
        print("RealtimeSTT initialized")
        recorder_ready.set()
        while True:
            full_sentence = recorder.text()
            if full_sentence:
                asyncio.new_event_loop().run_until_complete(
                    send_to_client(
                        json.dumps(
                            {"type": "fullSentence", "text": full_sentence},
                        ),
                    )
                )
                print(f"\rSentence: {full_sentence}")

                history.append(
                    {
                        "role": "user",
                        "content": full_sentence,
                    }
                )

                chat_completion = groq_client.chat.completions.create(
                    messages=history,
                    model="llama3-8b-8192",
                )
                print("Response", chat_completion.choices[0].message.content)

                if chat_completion.choices[0].message.content is None:
                    history.append(
                        {
                            "role": "assistant",
                            "content": "Failed to generate response",
                        }
                    )
                    continue

                llm_response = chat_completion.choices[0].message.content

                history.append(
                    {
                        "role": "assistant",
                        "content": llm_response,
                    }
                )

                # strip any markdown formatting, only keep alphanumeric characters, whitespaces, and punctuations
                llm_response = strip_markdown_and_special_chars(llm_response)

                print("Starting TTS...")
                tts.tts_to_file(
                    text=llm_response,
                    file_path="wav/audio.wav",
                    speaker="p287",
                )

                # read wav file and convert to base64 to send over websocket
                wav_base64 = base64.b64encode(
                    open("wav/audio.wav", "rb").read()
                ).decode("utf-8")
                print("TTS Completed")

                asyncio.new_event_loop().run_until_complete(
                    send_to_client(
                        json.dumps(
                            {
                                "type": "llm",
                                "text": chat_completion.choices[0].message.content,
                                "audio": "data:audio/wav;base64," + wav_base64,
                            },
                        ),
                    )
                )

    def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):

        # Decode 16-bit PCM data to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate the number of samples after resampling
        num_original_samples = len(audio_np)
        num_target_samples = int(
            num_original_samples * target_sample_rate / original_sample_rate
        )

        # Resample the audio
        resampled_audio = resample(audio_np, num_target_samples)

        return resampled_audio.astype(np.int16).tobytes()  # type: ignore

    async def echo(websocket):
        global current_uuid
        global client_websocket
        global history

        if client_websocket:
            await send_to_client(
                json.dumps(
                    {
                        "type": "disconnect",
                        "text": "Another client connected, disconnecting...",
                    },
                ),
            )
            print("Client disconnected", current_uuid)
            current_uuid = GetUUID()
        else:
            print("First client connected", current_uuid)

        print("Client connected", current_uuid)
        history = []

        client_websocket = websocket
        async for message in websocket:
            if not recorder_ready.is_set():
                print("Recorder not ready")
                continue

            metadata_length = int.from_bytes(message[:4], byteorder="little")
            metadata_json = message[4 : 4 + metadata_length].decode("utf-8")
            metadata = json.loads(metadata_json)
            sample_rate = metadata["sampleRate"]
            chunk = message[4 + metadata_length :]
            resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
            recorder.feed_audio(resampled_chunk)  # type: ignore

    # start_server = websockets.serve(echo, "0.0.0.0", 9001)
    start_server = websockets.serve(echo, "localhost", 8001)

    recorder_thread = threading.Thread(target=recorder_thread)  # type: ignore
    recorder_thread.start()  # type: ignore
    recorder_ready.wait()

    print("Server started. Press Ctrl+C to stop the server.")
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping server...")
        print("Server stopped.")
        exit(0)
