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
//   - Every request starts a *fresh* `codex exec` session, replaying the full conversation
//     flattened into one prompt, rather than using `codex exec resume`. The Chat Completions
//     client already resends the entire message history on every call, so this is simplest thing
//     that is correct: no session-id bookkeeping, no risk of resuming the wrong session when
//     multiple Workshop conversations are open at once. The tradeoff is no server-side session
//     reuse, which mainly costs a bit of latency, not correctness.
//   - Responses are NOT incrementally streamed token-by-token: this bridge waits for `codex exec`
//     to finish, then emits the full reply as a couple of SSE chunks (enough to satisfy Workshop's
//     always-streaming client). Real incremental streaming would mean parsing `codex exec --json`'s
//     internal event schema, which isn't documented stably enough to build on.
//   - Runs with `--sandbox read-only`: Codex can read the scratch directory it runs in but never
//     writes files or executes mutating commands. `codex exec` has no TTY here, so it already
//     defaults to `approval: never` in this mode (there's nothing to prompt).

import { createServer } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

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

// Run `codex exec` against a flattened prompt and resolve with its final reply text.
function runCodexExec(prompt) {
  return new Promise((resolve, reject) => {
    const outputPath = join(SCRATCH_DIR, `reply-${randomUUID()}.txt`);
    const child = spawn(args.codexBin, [
      "exec",
      "--json",
      "--skip-git-repo-check",
      "--sandbox", args.sandbox,
      "--output-last-message", outputPath,
      prompt,
    ], { cwd: SCRATCH_DIR, stdio: ["ignore", "pipe", "pipe"] });

    let stderr = "";
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    // Drain stdout (the --json event stream) without buffering it -- it's only used for the exit
    // code / stderr below, not parsed, per the module doc comment above.
    child.stdout.resume();

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
      try {
        const reply = readFileSync(outputPath, "utf8").trim();
        resolve(reply.length > 0 ? reply : "(codex exec returned no output)");
      } catch (error) {
        reject(new Error(`codex exec finished but no reply file was found: ${error.message}`));
      } finally {
        rmSync(outputPath, { force: true });
      }
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

// Emit a reply as a couple of OpenAI-style SSE `chat.completion.chunk` events -- not real
// token-by-token streaming (see module doc comment), just enough shape for pi-ai's streaming
// client, which requires at least one delta chunk plus a chunk carrying `finish_reason`.
function streamReply(res, model, text) {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const id = `chatcmpl-${randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  const chunk = (delta, finishReason = null) => JSON.stringify({
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  res.write(`data: ${chunk({ role: "assistant", content: text })}\n\n`);
  res.write(`data: ${chunk({}, "stop")}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
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

    const prompt = flattenMessages(request.messages);
    if (!prompt) {
      sendOpenAiError(res, 400, "No usable text content in messages");
      return;
    }

    console.log(`[codex-chat-bridge] ${new Date().toISOString()} prompt (${prompt.length} chars)`);
    try {
      const reply = await runCodexExec(prompt);
      if (request.stream === false) {
        sendNonStreamingReply(res, request.model ?? "codex", reply);
      } else {
        streamReply(res, request.model ?? "codex", reply);
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
