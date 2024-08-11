class AudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const inputChannel = input[0];
      const outputData = new Int16Array(inputChannel.length);

      for (let i = 0; i < inputChannel.length; i++) {
        outputData[i] = Math.max(
          -32768,
          Math.min(32767, inputChannel[i] * 32768),
        );
      }

      this.port.postMessage(outputData);
    }

    return true;
  }
}

registerProcessor("processor", AudioProcessor);
