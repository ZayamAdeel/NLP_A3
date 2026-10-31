// AudioWorklet: mic (browser sample rate) -> 16 kHz mono PCM16, posted in 100 ms frames.
// Each output sample is the mean of the input samples it spans (a cheap low-pass), which is
// good enough for speech recognition and handles non-integer ratios such as 44.1 kHz.
class Downsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.acc = 0;
    this.n = 0;
    this.buf = new Int16Array(1600); // 100 ms
    this.fill = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.n++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.n));
        this.buf[this.fill++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        this.acc = 0;
        this.n = 0;
        if (this.fill === this.buf.length) {
          this.port.postMessage(this.buf.slice().buffer);
          this.fill = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("downsampler", Downsampler);
