#!/usr/bin/env node

// Standalone local bridge that lets Workshop's AI chat talk to your OpenAI Codex CLI subscription
// instead of a metered provider API key. This is NOT part of the Workers backend -- it's a plain
// Node process you run alongside `pnpm dev-server`, and it exposes an OpenAI-Chat-Completions-
// compatible endpoint (POST /v1/chat/completions) backed by `codex exec`.
//
// Why this exists: Cloudflare Workers (even under `wrangler dev`) run inside workerd, which has no
// subprocess/child_process support, so nothing running in the Worker itself can ever shell out to
// `codex`. This script is the real OS process that does the exec'ing; the Worker just fetches it
// like any other OpenAI-compatible server (the same path Workshop already supports for Ollama).
//
// Setup:
//   1. Install & log in to the Codex CLI (npm i -g @openai/codex, then `codex login`).
//   2. Run this bridge:  node scripts/codex-chat-bridge.mjs
//   3. In Workshop, Add Model -> provider "Ollama" (its "no API key" path fits this best) ->
//      "Other Ollama..." -> Model ID: whatever you like (e.g. "codex") -> API URL:
//      http://localhost:4321  (or your --port) -> leave API Token blank.
//
// Scope & limitations (see plans/pi-impl.md and packages/workshop-backend/src/ai-models.ts for the
// consumer side of this contract):
//   - Text-only conversational chat. `codex exec` is an agentic coding tool, not a chat-completion
//     function -- there's no reliable way to make it honor Workshop's structured tool-calling
//     contract, so this bridge does not support gadget-building agent flows, only the plain AI
//     chat feature. Tool/tool-result messages in the incoming history are rendered as inert text
//     markers so requests don't crash, not executed as real tool calls.
//   - Session resume: Chat Completions has no session concept of its own -- the client just resends
//     the entire message history on every call. To still use `codex exec resume` (avoiding a cold
//     CLI start on every turn), each reply is indexed by a content hash of "the history that led to
//     it" (see conversationKey/sessions below); the next request's history-minus-its-last-message is
//     hashed and looked up against that index. A hit means "this is the same conversation
//     continuing" -> resume with just the new message. A miss (new conversation, or an earlier
//     message was edited/regenerated) -> start fresh, replaying the whole history as one prompt. If
//     a resume attempt fails outright (e.g. a stale/evicted session) before any output was sent to
//     the client, it's retried once as a fresh session rather than failing the request.
//   - Streaming is real but segment-level, not token-level: `codex exec --json` never emits partial
//     tokens -- each agent_message only ever appears as one complete `item.completed` event. This
//     bridge forwards each such segment to the client as soon as it arrives (so a multi-step reply
//     appears in pieces as Codex produces them), but a single-segment reply still appears all at
//     once, same as it would talking to Codex directly.
//   - Runs with `--sandbox read-only`: Codex can read the scratch directory it runs in but never
//     writes files or executes mutating commands. `codex exec` has no TTY here, so it already
//     defaults to `approval: never` in this mode (there's nothing to prompt). Note `codex exec
//     resume` has no `--sandbox` flag of its own -- a resumed session keeps the sandbox mode it was
//     first created with.

import { createServer } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID, createHash } from "node:crypto";

function parseArgs(argv) {
  const args = { port: 4321, codexBin: "codex", sandbox: "read-only", timeoutMs: 300_000 };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === "--port") args.port = Number(argv[++i]);
    else if (flag === "--codex-bin") args.codexBin = argv[++i];
    else if (flag === "--sandbox") args.sandbox = argv[++i];
    else if (flag === "--timeout-ms") args.timeoutMs = Number(argv[++i]);
  }
  return args;
}
const args = parseArgs(process.argv.slice(2));

// A single scratch directory the bridge always runs `codex exec` from. `--skip-git-repo-check`
// means it doesn't need to be a real git repo; keeping it separate from this actual repo means
// Codex never sees (or could touch) this project's own files while answering chat messages.
const SCRATCH_DIR = mkdtempSync(join(tmpdir(), "codex-chat-bridge-"));

// conversationKey(history through the assistant's reply) -> { threadId }. Grows for the life of the
// process; fine for personal local use (a handful of conversations at a time), not meant to bound.
const sessions = new Map();

function checkCodexAvailable() {
  const result = spawnSync(args.codexBin, ["--version"], { stdio: "pipe" });
  if (result.error || result.status !== 0) {
    console.error(
      `warning: could not run "${args.codexBin} --version" -- is the Codex CLI installed and ` +
      `on PATH? (npm i -g @openai/codex, then \`codex login\`). Requests will fail until it is.`
    );
  }
}

// Flatten an OpenAI Chat Completions `messages` array into a single transcript prompt, since
// `codex exec` takes one prompt string, not a role-tagged message array. Only text is preserved;
// image parts and tool_calls/tool-result messages become inert text markers.
function flattenMessages(messages) {
  const lines = [];
  for (const message of messages ?? []) {
    const role = message.role === "developer" ? "system" : message.role;
    const label = role === "user" ? "User" : role === "assistant" ? "Assistant"
        : role === "tool" ? "Tool result" : "System";
    const text = extractText(message);
    if (text) lines.push(`[${label}]\n${text}`);
  }
  return lines.join("\n\n");
}

function extractText(message) {
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
        .map((part) => part?.type === "text" ? part.text : part?.type ? `(${part.type} omitted)` : "")
        .filter(Boolean)
        .join("\n");
  }
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    return message.tool_calls
        .map((call) => `(requested tool call: ${call.function?.name ?? call.custom?.name ?? "unknown"})`)
        .join("\n");
  }
  return "";
}

// Deterministic key for a message history, used to recognize "this request is the same
// conversation as a reply we gave before, plus one new message" without the client sending any
// session id of its own (see module doc comment).
function conversationKey(messages) {
  const hash = createHash("sha256");
  for (const message of messages) {
    hash.update(message.role ?? "");
    hash.update("\0");
    hash.update(extractText(message));
    hash.update("\0");
  }
  return hash.digest("hex");
}

// Run `codex exec` (fresh, or resuming `resumeThreadId`), calling `onSegment(text)` for each
// agent_message as it completes. Resolves with { threadId, fullText } once the turn finishes.
function runCodex({ prompt, resumeThreadId, onSegment }) {
  return new Promise((resolve, reject) => {
    const cliArgs = resumeThreadId
        ? ["exec", "resume", resumeThreadId, "--json", "--skip-git-repo-check", prompt]
        : ["exec", "--json", "--skip-git-repo-check", "--sandbox", args.sandbox, prompt];
    const child = spawn(args.codexBin, cliArgs, { cwd: SCRATCH_DIR, stdio: ["ignore", "pipe", "pipe"] });

    let stderr = "";
    let buffer = "";
    let threadId = resumeThreadId ?? null;
    const segments = [];

    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try { event = JSON.parse(line); } catch { continue; }
        if (event.type === "thread.started" && event.thread_id) {
          threadId = event.thread_id;
        } else if (event.type === "item.completed" && event.item?.type === "agent_message"
            && typeof event.item.text === "string") {
          segments.push(event.item.text);
          onSegment(event.item.text);
        }
      }
    });
    child.stderr.on("data", (chunk) => { stderr += chunk; });

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`codex exec timed out after ${args.timeoutMs}ms`));
    }, args.timeoutMs);

    child.on("error", (error) => {
      clearTimeout(timer);
      reject(new Error(`failed to spawn "${args.codexBin}": ${error.message}`));
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`codex exec exited with code ${code}${stderr ? `: ${stderr.trim()}` : ""}`));
        return;
      }
      if (!threadId) {
        reject(new Error("codex exec finished without reporting a thread id"));
        return;
      }
      const fullText = segments.join("\n\n").trim();
      resolve({ threadId, fullText: fullText.length > 0 ? fullText : "(codex exec returned no output)" });
    });
  });
}

function sendJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(payload) });
  res.end(payload);
}

function sendOpenAiError(res, status, message) {
  sendJson(res, status, { error: { message, type: "codex_bridge_error" } });
}

function sendNonStreamingReply(res, model, text) {
  sendJson(res, 200, {
    id: `chatcmpl-${randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
  });
}

const server = createServer((req, res) => {
  if (req.method !== "POST" || req.url.split("?")[0] !== "/v1/chat/completions") {
    sendOpenAiError(res, 404, `Not found: ${req.method} ${req.url}`);
    return;
  }

  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", async () => {
    let request;
    try {
      request = JSON.parse(body);
    } catch {
      sendOpenAiError(res, 400, "Invalid JSON body");
      return;
    }

    const messages = request.messages ?? [];
    if (messages.length === 0) {
      sendOpenAiError(res, 400, "No messages in request");
      return;
    }

    const model = request.model ?? "codex";
    const streaming = request.stream !== false;
    const priorKey = conversationKey(messages.slice(0, -1));
    let existing = sessions.get(priorKey);
    let prompt = existing ? extractText(messages[messages.length - 1]) : flattenMessages(messages);
    if (!prompt) {
      sendOpenAiError(res, 400, "No usable text content in messages");
      return;
    }

    // SSE headers/id are only opened on the first segment, so a same-request retry-as-fresh (see
    // below) that hasn't emitted anything yet can still cleanly fall back without corrupting the
    // response; once opened, we're committed to this stream.
    let sseStarted = false;
    let sseId, sseCreated;
    const emitSegment = (text) => {
      if (!streaming) return;
      if (!sseStarted) {
        sseStarted = true;
        res.writeHead(200, {
          "content-type": "text/event-stream",
          "cache-control": "no-cache",
          connection: "keep-alive",
        });
        sseId = `chatcmpl-${randomUUID()}`;
        sseCreated = Math.floor(Date.now() / 1000);
      }
      const chunk = JSON.stringify({
        id: sseId, object: "chat.completion.chunk", created: sseCreated, model,
        choices: [{ index: 0, delta: { role: "assistant", content: `${text}\n\n` }, finish_reason: null }],
      });
      res.write(`data: ${chunk}\n\n`);
    };

    console.log(
      `[codex-chat-bridge] ${new Date().toISOString()} ${existing ? "resume" : "fresh"} ` +
      `prompt (${prompt.length} chars)`
    );
    try {
      let result;
      try {
        result = await runCodex({ prompt, resumeThreadId: existing?.threadId, onSegment: emitSegment });
      } catch (resumeError) {
        if (!existing || sseStarted) throw resumeError;
        console.error(`[codex-chat-bridge] resume failed (${resumeError.message}); retrying fresh`);
        existing = undefined;
        prompt = flattenMessages(messages);
        result = await runCodex({ prompt, resumeThreadId: undefined, onSegment: emitSegment });
      }

      const { threadId, fullText } = result;
      sessions.set(conversationKey([...messages, { role: "assistant", content: fullText }]), { threadId });

      if (streaming) {
        if (!sseStarted) emitSegment(fullText); // defensive: turn produced no agent_message segments
        const finishChunk = JSON.stringify({
          id: sseId, object: "chat.completion.chunk", created: sseCreated, model,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        });
        res.write(`data: ${finishChunk}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      } else {
        sendNonStreamingReply(res, model, fullText);
      }
    } catch (error) {
      console.error(`[codex-chat-bridge] error: ${error.message}`);
      if (!res.headersSent) sendOpenAiError(res, 500, error.message);
      else res.end();
    }
  });
});

checkCodexAvailable();
server.listen(args.port, () => {
  console.log(`codex-chat-bridge listening on http://localhost:${args.port}`);
  console.log(`scratch dir: ${SCRATCH_DIR}`);
  console.log(
    `In Workshop's Add Model dialog: provider "Ollama" -> "Other Ollama..." -> ` +
    `API URL http://localhost:${args.port} -> leave API Token blank.`
  );
});

process.on("SIGINT", () => {
  rmSync(SCRATCH_DIR, { recursive: true, force: true });
  process.exit(0);
});
