// Hermes lacks TextDecoder and the Float16 DataView methods; both are needed
// by the DepthART bundle loader (manifest JSON decode + fp16 weight upload).
// Load-time only - nothing on the per-frame path touches these.
import { Float16Array as F16Array, getFloat16, setFloat16 } from '@petamoriken/float16'

const g = globalThis as unknown as Record<string, unknown>

if (typeof g.TextDecoder === 'undefined') {
  class TextDecoderPolyfill {
    decode(input?: ArrayBuffer | ArrayBufferView): string {
      if (input == null) return ''
      const bytes =
        input instanceof Uint8Array
          ? input
          : ArrayBuffer.isView(input)
            ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
            : new Uint8Array(input)
      // UTF-8 decode (the manifest is JSON).
      let out = ''
      let i = 0
      while (i < bytes.length) {
        const b = bytes[i]!
        let cp: number
        if (b < 0x80) {
          cp = b
          i += 1
        } else if (b < 0xe0) {
          cp = ((b & 0x1f) << 6) | (bytes[i + 1]! & 0x3f)
          i += 2
        } else if (b < 0xf0) {
          cp = ((b & 0x0f) << 12) | ((bytes[i + 1]! & 0x3f) << 6) | (bytes[i + 2]! & 0x3f)
          i += 3
        } else {
          cp =
            ((b & 0x07) << 18) |
            ((bytes[i + 1]! & 0x3f) << 12) |
            ((bytes[i + 2]! & 0x3f) << 6) |
            (bytes[i + 3]! & 0x3f)
          i += 4
        }
        out += String.fromCodePoint(cp)
      }
      return out
    }
  }
  g.TextDecoder = TextDecoderPolyfill
}

const dvProto = DataView.prototype as unknown as Record<string, unknown>
if (typeof dvProto.getFloat16 === 'undefined') {
  dvProto.getFloat16 = function (this: DataView, offset: number, littleEndian?: boolean) {
    return getFloat16(this, offset, littleEndian)
  }
  dvProto.setFloat16 = function (
    this: DataView,
    offset: number,
    value: number,
    littleEndian?: boolean,
  ) {
    setFloat16(this, offset, value, littleEndian)
  }
}
if (typeof g.Float16Array === 'undefined') {
  g.Float16Array = F16Array
}
