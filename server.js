// server.js
// Remagine backend — one Groq API key, 8 labeled "AI" personas + 1 summary call.
// Minimal, single-file backend: static frontend + a small JSON API + in-memory history.

const express = require("express");
const path = require("path");
const crypto = require("crypto");

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

// One real model backs every persona. Swap this if Groq deprecates it —
// see console.groq.com/docs/models for the current list.
const GROQ_MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-20b";

if (!GROQ_API_KEY) {
  console.warn(
    "[warn] GROQ_API_KEY is not set. /api/ask will return a 500 until it is configured."
  );
} else {
  console.log(
    `[info] GROQ_API_KEY loaded (starts with "${GROQ_API_KEY.slice(0, 6)}...", length ${GROQ_API_KEY.length}). Using model "${GROQ_MODEL}".`
  );
}

// ---------------------------------------------------------------------------
// The 8 "AI" personas shown in the UI. Only one API key/model powers all of
// them — each persona is just a different system prompt + display color, so
// the 8 cards read like 8 different assistants without needing 8 accounts.
// ---------------------------------------------------------------------------
const PERSONAS = [
  {
    id: "perplexity",
    name: "Perplexity",
    color: "border-brandMint",
    system:
      "You are a research-first assistant, similar in spirit to Perplexity. Answer the user's question directly, then briefly mention what kind of source or evidence would confirm it. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "chatgpt",
    name: "ChatGPT",
    color: "border-[#86EFAC]",
    system:
      "You are a friendly, conversational assistant, similar in spirit to ChatGPT. Give a direct, clear answer first, then briefly offer to go deeper. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "gemini",
    name: "Gemini",
    color: "border-brandYellow",
    system:
      "You are a practical, action-oriented assistant, similar in spirit to Gemini. Lead with the most actionable next step, then explain briefly. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "claude",
    name: "Claude",
    color: "border-brandCoral",
    system:
      "You are a thorough, careful assistant, similar in spirit to Claude. Briefly lay out the key trade-offs or reasoning, then give your answer. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "grok",
    name: "Grok",
    color: "border-[#A78BFA]",
    system:
      "You are a candid, plainspoken assistant, similar in spirit to Grok. State your view directly and briefly, without hedging much. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "copilot",
    name: "Copilot",
    color: "border-brandBlue",
    system:
      "You are a task-focused assistant, similar in spirit to Copilot. Turn the answer into a concrete next step or checklist item the user can do right away. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "kimi",
    name: "Kimi",
    color: "border-[#FDBA74]",
    system:
      "You are a patient, detail-tolerant assistant, similar in spirit to Kimi. Feel free to consider more context and nuance than usual before answering. Keep it to 3-5 sentences. Do not mention you are an AI model or name any company.",
  },
  {
    id: "metaai",
    name: "Meta AI",
    color: "border-brandPink",
    system:
      "You are a casual, easy-to-skim assistant, similar in spirit to Meta AI. Keep the answer short, plain, and friendly. Keep it to 2-4 sentences. Do not mention you are an AI model or name any company.",
  },
];

// ---------------------------------------------------------------------------
// In-memory history store. Fine for a demo/single-instance deploy; swap for a
// real database if you need history to survive restarts or scale to >1 dyno.
// ---------------------------------------------------------------------------
const sessions = new Map(); // id -> { id, question, createdAt, answers: [{id,name,color,text}], summary }
const HISTORY_LIMIT = 200;

function pushSession(session) {
  sessions.set(session.id, session);
  if (sessions.size > HISTORY_LIMIT) {
    const oldestKey = sessions.keys().next().value;
    sessions.delete(oldestKey);
  }
}

async function callGroq(messages, { maxTokens = 1400, temperature = 0.8, jsonSchema = null } = {}) {
  const body = {
    model: GROQ_MODEL,
    messages,
    temperature,
    max_completion_tokens: maxTokens,
  };
  if (jsonSchema) {
    body.response_format = {
      type: "json_schema",
      json_schema: jsonSchema,
    };
  }

  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Groq request failed (${res.status}): ${errText.slice(0, 500)}`);
  }

  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error("Groq returned an empty response");
  return text;
}

// ---------------------------------------------------------------------------
// Single-request mode: one Groq call asks the model to role-play all 8
// personas AND write the summary, returned as one JSON object. This uses
// ~1 request instead of 9, which is what actually matters for Groq's
// per-minute/per-day request limits (the token count is similar either way,
// since the model still has to generate roughly the same amount of text).
//
// We use Groq's structured-outputs mode (response_format: json_schema,
// strict: true) rather than the looser json_object mode. gpt-oss-20b
// supports strict schema-constrained decoding, which guarantees the shape
// is valid — json_object mode only asks nicely via the prompt and can
// reject the whole request with a 400 "Failed to validate JSON" error if
// the model's free-form attempt doesn't parse.
// ---------------------------------------------------------------------------
function buildPersonaSchema(personas) {
  const answerProps = {};
  personas.forEach((p) => {
    answerProps[p.id] = { type: "string", description: `${p.name}'s answer, in that persona's style.` };
  });

  return {
    name: "persona_answers",
    strict: true,
    schema: {
      type: "object",
      properties: {
        answers: {
          type: "object",
          properties: answerProps,
          required: personas.map((p) => p.id),
          additionalProperties: false,
        },
        summary: {
          type: "string",
          description: "2-4 sentence neutral summary comparing the persona answers.",
        },
      },
      required: ["answers", "summary"],
      additionalProperties: false,
    },
  };
}

function buildPrompt(question, personas) {
  const roster = personas
    .map((p, i) => `${i + 1}. id: "${p.id}", label: "${p.name}" — style: ${p.system}`)
    .join("\n");

  return (
    `You will answer one user question from ${personas.length} distinct assistant personas, ` +
    `then write a short summary comparing them. Stay fully in character for each persona and ` +
    `make sure the answers genuinely differ in angle, tone and emphasis — do not repeat the same ` +
    `sentence structure across personas.\n\n` +
    `Personas:\n${roster}\n\n` +
    `User question: ${question}\n\n` +
    `Rules:\n` +
    `- Each persona answer: 2-4 sentences, in that persona's style, answering the question directly.\n` +
    `- Never mention you are an AI language model or name any real company/product in the answers.\n` +
    `- The summary must be 2-4 sentences: note where the personas broadly agree, name any real ` +
    `disagreement, and end with a plain-language takeaway. Do not just restate every answer.`
  );
}

function extractJson(raw) {
  // Structured-output mode should already return clean JSON, but keep a
  // defensive fallback in case a fence or stray text slips through.
  let cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  try {
    return JSON.parse(cleaned);
  } catch (_err) {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) throw new Error("No JSON object found in model output");
    return JSON.parse(cleaned.slice(start, end + 1));
  }
}

async function askAllPersonasOneShot(question, selectedIds) {
  const personas = selectedIds && selectedIds.length
    ? PERSONAS.filter((p) => selectedIds.includes(p.id))
    : PERSONAS;

  const prompt = buildPrompt(question, personas);
  const schema = buildPersonaSchema(personas);

  let parsed;
  try {
    const raw = await callGroq(
      [
        {
          role: "system",
          content: "You answer questions from multiple assistant personas at once and summarize them, always following the given response schema exactly.",
        },
        { role: "user", content: prompt },
      ],
      { maxTokens: 1600, temperature: 0.8, jsonSchema: schema }
    );
    parsed = extractJson(raw);
  } catch (err) {
    console.error("[askAllPersonasOneShot] Groq call/parse failed:", err.message || err);
    // Whole call (or JSON parse) failed — fall back to a uniform error state
    // for every persona rather than partial/broken data. The real error is
    // surfaced in the summary text itself so it's visible on the chat page,
    // not just in server logs — makes misconfiguration (bad key, wrong model
    // id, rate limit) diagnosable without needing to check Render's logs.
    const message = String(err.message || err);
    const answers = personas.map((p) => ({
      id: p.id,
      name: p.name,
      color: p.color,
      text: "This model couldn't answer right now.",
      ok: false,
      error: message,
    }));
    return {
      answers,
      summary: `Something went wrong talking to the model, so no answers came back. Details: ${message}`,
    };
  }

  const answers = personas.map((p) => {
    const text = typeof parsed?.answers?.[p.id] === "string" ? parsed.answers[p.id].trim() : "";
    return {
      id: p.id,
      name: p.name,
      color: p.color,
      text: text || "This model couldn't answer right now.",
      ok: Boolean(text),
    };
  });

  const summary =
    typeof parsed?.summary === "string" && parsed.summary.trim()
      ? parsed.summary.trim()
      : "Summary unavailable right now, but you can compare the answers above directly.";

  return { answers, summary };
}

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------

app.get("/api/models", (_req, res) => {
  res.json({ personas: PERSONAS.map(({ id, name, color }) => ({ id, name, color })) });
});

app.post("/api/ask", async (req, res) => {
  try {
    const question = String(req.body?.question || "").trim();
    const selectedIds = Array.isArray(req.body?.models) ? req.body.models : null;

    if (!question) {
      return res.status(400).json({ error: "Question is required." });
    }
    if (!GROQ_API_KEY) {
      return res.status(500).json({ error: "Server is missing GROQ_API_KEY." });
    }

    const { answers, summary } = await askAllPersonasOneShot(question, selectedIds);

    const session = {
      id: crypto.randomBytes(6).toString("hex"),
      question,
      createdAt: new Date().toISOString(),
      answers,
      summary,
    };
    pushSession(session);

    res.json(session);
  } catch (err) {
    console.error("/api/ask error:", err);
    res.status(500).json({ error: "Something went wrong asking the models." });
  }
});

app.get("/api/session/:id", (req, res) => {
  const session = sessions.get(req.params.id);
  if (!session) return res.status(404).json({ error: "Session not found." });
  res.json(session);
});

app.get("/api/history", (_req, res) => {
  const list = Array.from(sessions.values())
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    .map((s) => ({ id: s.id, question: s.question, createdAt: s.createdAt }));
  res.json({ sessions: list });
});

// Separate pages (chat/result view + all-chats view) live as static files
// in /public and are served automatically by express.static above:
//   /chat.html?id=<sessionId>   -> one question + 8 answers + summary
//   /history.html               -> list of all past chats

app.listen(PORT, () => {
  console.log(`Remagine server listening on port ${PORT}`);
});
