// Attach to the app's Hermes runtime via Metro's CDP proxy.
// Usage:
//   node scripts/jsconsole.mjs [seconds]            - stream console output
//   node scripts/jsconsole.mjs [seconds] "expr"     - also evaluate an expression
const seconds = Number(process.argv[2] ?? 10)
const expression = process.argv[3]

const targets = await (await fetch('http://localhost:8081/json')).json()
const target = targets.find((t) => t.appId?.includes('lightdemo')) ?? targets[0]
if (!target) {
  console.error('no inspector target')
  process.exit(1)
}
import { WebSocket as WSClient } from 'ws'
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

const fmt = (arg) => {
  if (arg.type === 'string') return arg.value
  if (arg.value !== undefined) return JSON.stringify(arg.value)
  if (arg.description !== undefined) return arg.description
  return `[${arg.type}]`
}

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data)
  if (msg.id != null) {
    pending.get(msg.id)?.(msg.result ?? msg.error)
    pending.delete(msg.id)
    return
  }
  if (msg.method === 'Runtime.consoleAPICalled') {
    const { type, args } = msg.params
    console.log(`[${type}]`, args.map(fmt).join(' '))
  }
  if (msg.method === 'Runtime.exceptionThrown') {
    console.log('[EXCEPTION]', JSON.stringify(msg.params.exceptionDetails).slice(0, 800))
  }
}

ws.onerror = (e) => console.error('[WS ERROR]', e.message ?? e)
ws.onclose = (e) => console.error('[WS CLOSED]', e.code, e.reason)
ws.onopen = async () => {
  console.error('[WS OPEN]', target.webSocketDebuggerUrl)
  await send('Runtime.enable')
  if (expression) {
    const result = await send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    })
    console.log('[EVAL]', JSON.stringify(result, null, 1).slice(0, 4000))
  }
}
setTimeout(() => process.exit(0), seconds * 1000)
