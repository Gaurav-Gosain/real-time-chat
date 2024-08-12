# Real-Time Chat Application

This repository contains a real-time chat application that integrates state-of-the-art technologies to provide seamless speech-to-text (STT), language processing, and text-to-speech (TTS) functionalities. The application is designed to allow users to communicate through audio, with messages being transcribed, processed, and spoken back in real-time.

## Features

- **Speech-to-Text (STT)**: Utilizes FastWhisper to convert spoken audio into text in real-time.
- **Language Model Processing**: Employs Groq to generate intelligent responses based on user input.
- **Text-to-Speech (TTS)**: Converts the generated text response into speech using Coqui TTS.
- **Real-Time Communication**: Audio data is streamed via WebSockets from the frontend to the Python backend.
- **Frontend UI**: Built using Next.js, Shadcn/UI, and TailwindCSS for a modern and responsive user interface.

## Technologies

### Backend

- **Python**
  - `websockets`: For handling real-time communication between the client and server.
  - `FastWhisper`: Provides real-time speech-to-text transcription.
  - `RealtimeSTT`: Real-time STT functionality using FastWhisper.
  - `Groq`: Language model API used for generating responses.
  - `Coqui TTS`: Converts text responses into speech.

### Frontend

- **Next.js**: A React framework for building server-rendered applications.
- **Shadcn/UI**: A UI framework built on TailwindCSS for modern and accessible component design.
- **TailwindCSS**: A utility-first CSS framework for creating responsive designs.

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- WebSockets library
- Required Python packages (see `pyproject.toml`)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/real-time-chat-app.git
   cd real-time-chat-app
   ```

2. **Install Backend Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies:**

   ```bash
   cd frontend
   npm install
   ```

4. **Set up environment variables:**

   - Create a `.env` file in the root directory and add your `GROQ_API_KEY`.

5. **Start the Backend Server:**

   ```bash
   python backend/server.py
   ```

6. **Start the Frontend Development Server:**

   ```bash
   cd frontend
   npm run dev
   ```

7. **Access the Application:**

   Open your browser and navigate to `http://localhost:3000`.

## Usage

- **Connecting:** Once the frontend is loaded, you can start speaking into your microphone. The audio will be sent to the backend server, where it will be transcribed, processed, and converted back to audio.
- **Real-Time Feedback:** The frontend will display the transcribed text and play back the TTS-generated audio response.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## Future Work

- Use deepgram Whisper API for faster transcription.
- Use ElevenLabs API for faster transcription.
- Add support for multiple concurrent connections.
- Authentication/chat persistence.
- Add support for other LLMs.

## License

This project is licensed under the MIT License.
