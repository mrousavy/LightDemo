// Persistent CDP session: with worklets bundleMode, Hermes' debugger agent
// segfaults on attach/detach cycling - so attach ONCE and never let go.
// Console output streams to stdout; evals are read from a queue file
// (scratch/evalq.txt): write JS to it and the result prints here.
import { readFileSync, watchFile, writeFileSync } from 'node:fs'
import { WebSocket as WSClient } from 'ws'

const QUEUE = process.argv[2] ?? '/tmp/evalq.txt'
writeFileSync(QUEUE, '')

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
    console.log(`[${msg.params.type}]`, msg.params.args.map(fmt).join(' '))
  }
}
ws.onopen = async () => {
  await send('Runtime.enable')
  console.log('[daemon] attached')
  let last = ''
  watchFile(QUEUE, { interval: 300 }, async () => {
    let expr = ''
    try {
      expr = readFileSync(QUEUE, 'utf8').trim()
    } catch {}
    if (expr === '' || expr === last) return
    last = expr
    const result = await send('Runtime.evaluate', {
      expression: expr,
      returnByValue: false,
      generatePreview: false,
    })
    console.log('[EVAL-RESULT]', JSON.stringify(result?.result ?? result).slice(0, 2000))
  })
}
ws.onclose = () => {
  console.log('[daemon] socket closed')
  process.exit(0)
}
