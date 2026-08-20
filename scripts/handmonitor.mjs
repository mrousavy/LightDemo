// Watch hand-tracking status live and archive snapshots while a hand is
// tracked. Usage: node scripts/handmonitor.mjs [seconds] [snapshotDir]
import { WebSocket as WSClient } from 'ws'
import { copyFileSync, existsSync } from 'node:fs'

const seconds = Number(process.argv[2] ?? 300)
const snapDir = process.argv[3] ?? '.'
const containerSnap =
  process.env.HOME +
  '/Library/Containers/367C941F-7B3C-4694-A0AA-584084EC71B4/Data/Documents/snap.png'

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

const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true })
  return r?.result?.value
}

let wasTracked = false
let wasGrabbed = false
let lastSnapAt = 0
let snapCount = 0

ws.on('open', async () => {
  await send('Runtime.enable')
  console.log('monitoring hand status...')
  const interval = setInterval(async () => {
    const json = await evaluate(
      'JSON.stringify(globalThis.__nitro ? globalThis.__nitro.getStatus() : null)',
    )
    if (json == null) return
    const s = JSON.parse(json)
    if (s == null) return
    const tracked = s.handTracked
    const grabbed = s.grabbed
    if (tracked !== wasTracked || grabbed !== wasGrabbed) {
      console.log(
        `${new Date().toISOString()} tracked=${tracked} grabbed=${grabbed} ` +
          `pinch=${s.pinchRatio.toFixed(2)} light=(${s.lightX.toFixed(2)},` +
          `${s.lightY.toFixed(2)},${s.lightZ.toFixed(2)}) fps=${s.fps.toFixed(0)}`,
      )
      wasTracked = tracked
      wasGrabbed = grabbed
    }
    if (tracked) {
      console.log(
        `  hand pinch=${s.pinchRatio.toFixed(2)} grabbed=${grabbed} ` +
          `light=(${s.lightX.toFixed(2)},${s.lightY.toFixed(2)},${s.lightZ.toFixed(2)}) ` +
          `depth=${s.depthTimeMs.toFixed(0)}ms hand=${s.handTimeMs.toFixed(0)}ms`,
      )
      const now = Date.now()
      if (now - lastSnapAt > 3200 && existsSync(containerSnap)) {
        lastSnapAt = now
        snapCount++
        copyFileSync(containerSnap, `${snapDir}/hand-${String(snapCount).padStart(2, '0')}.png`)
        console.log(`  saved hand-${String(snapCount).padStart(2, '0')}.png`)
      }
    }
  }, 400)
  setTimeout(() => {
    clearInterval(interval)
    process.exit(0)
  }, seconds * 1000)
})
