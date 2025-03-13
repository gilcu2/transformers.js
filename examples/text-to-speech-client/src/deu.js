import { pipeline } from '@huggingface/transformers';

// Create a text-to-speech pipeline
const synthesizer = await pipeline('text-to-speech', 'Xenova/mms-tts-deu', {
    quantized: false, // Remove this line to use the quantized version (default)
});

// Generate speech
const output = await synthesizer('Hallo');
console.log(output);
// {
//   audio: Float32Array(18432) [ ... ],
//   sampling_rate: 16000
// }
