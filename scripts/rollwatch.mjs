// Watch the face-roll calibration + hand status live.
import { WebSocket as WSClient } from 'ws'

const seconds = Number(process.argv[2] ?? 600)
const targets = await (await fetch('http://localhost:8081/json')).json()
const target = targets.find((t) => t.appId?.includes('lightdemo')) ?? targets[0]
if (!target) {
  console.error('no inspector target')
  process.exit(1)
}
const ws = new WSClient(target.webSocketDebuggerUrl, {
  headers: { Origin: 'http://localhost:8081' },
})
let id = 1
const pending = new Map()
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const msgId = id++
    pending.set(msgId, resolve)
    ws.send(JSON.stringify({ id: msgId, method, params }))
  })
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString())
  if (msg.id != null) {
    pending.get(msg.id)?.(msg.result ?? msg.error)
    pending.delete(msg.id)
  }
})
let lastRoll = null
let lastTracked = null
let lastGrabbed = null
ws.on('open', async () => {
  await send('Runtime.enable')
  console.log('watching roll + hand...')
  const interval = setInterval(async () => {
    const r = await send('Runtime.evaluate', {
      expression:
        'JSON.stringify(globalThis.__nitro ? {roll: globalThis.__nitro.lastFaceRollDegrees, s: globalThis.__nitro.getStatus()} : null)',
      returnByValue: true,
    })
    const json = r?.result?.value
    if (json == null) return
    const data = JSON.parse(json)
    if (data == null) return
    const roll = Math.round(data.roll)
    const { handTracked, grabbed, lightX, lightY, lightZ, pinchRatio } = data.s
    if (roll !== lastRoll || handTracked !== lastTracked || grabbed !== lastGrabbed) {
      const mapA = ((Math.round(-roll / 90) * 90) % 360 + 360) % 360
      const mapB = ((Math.round(roll / 90) * 90) % 360 + 360) % 360
      console.log(
        `${new Date().toISOString()} roll=${roll} (sign-1->rot ${mapA}, sign+1->rot ${mapB}) ` +
          `tracked=${handTracked} grabbed=${grabbed} pinch=${pinchRatio.toFixed(2)} ` +
          `light=(${lightX.toFixed(2)},${lightY.toFixed(2)},${lightZ.toFixed(2)})`,
      )
      lastRoll = roll
      lastTracked = handTracked
      lastGrabbed = grabbed
    }
  }, 1500)
  setTimeout(() => {
    clearInterval(interval)
    process.exit(0)
  }, seconds * 1000)
})
