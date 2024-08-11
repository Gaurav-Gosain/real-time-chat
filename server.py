if __name__ == "__main__":
    print("Starting server, please wait...")

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

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    print(TTS().list_models())

    # Init TTS
    # tts = TTS("vocoder_models/en/sam/hifigan_v2").to(device)
    tts = TTS("tts_models/en/vctk/vits").to(device)

    recorder = None
    recorder_ready = threading.Event()
    client_websocket = None

    history: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep your responses as short and concise as possible.",
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

                history.append(
                    {
                        "role": "assistant",
                        "content": chat_completion.choices[0].message.content,
                    }
                )

                print("Starting TTS...")
                tts.tts_to_file(
                    # speaker="Suad Qasim",
                    # speed=0.2,
                    # language="ar",
                    text=chat_completion.choices[0].message.content,
                    file_path=f"wav/{len(history)}.wav",
                    speaker="p287",
                )
                print("TTS Completed")

                asyncio.new_event_loop().run_until_complete(
                    send_to_client(
                        json.dumps(
                            {
                                "type": "llm",
                                "text": chat_completion.choices[0].message.content,
                                "audio": f"{os.getenv('REALTIME_SERVER_URL')}/{len(history)}.wav",
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

    async def echo(websocket, path):
        print("Client connected")
        global client_websocket
        global history
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
            recorder.feed_audio(resampled_chunk)

    # start_server = websockets.serve(echo, "0.0.0.0", 9001)
    start_server = websockets.serve(echo, "localhost", 8001)

    recorder_thread = threading.Thread(target=recorder_thread)
    recorder_thread.start()
    recorder_ready.wait()

    print("Server started. Press Ctrl+C to stop the server.")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
