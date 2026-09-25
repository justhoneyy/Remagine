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

async function callGroq(systemPrompt, userQuestion, { maxTokens = 220, temperature = 0.7 } = {}) {
  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userQuestion },
      ],
      temperature,
      max_completion_tokens: maxTokens,
    }),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Groq request failed (${res.status}): ${errText.slice(0, 300)}`);
  }

  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error("Groq returned an empty response");
  return text;
}

async function askAllPersonas(question, selectedIds) {
  const personas = selectedIds && selectedIds.length
    ? PERSONAS.filter((p) => selectedIds.includes(p.id))
    : PERSONAS;

  const results = await Promise.all(
    personas.map(async (persona) => {
      try {
        const text = await callGroq(persona.system, question, { maxTokens: 180 });
        return { id: persona.id, name: persona.name, color: persona.color, text, ok: true };
      } catch (err) {
        return {
          id: persona.id,
          name: persona.name,
          color: persona.color,
          text: "This model couldn't answer right now.",
          ok: false,
          error: String(err.message || err),
        };
      }
    })
  );
  return results;
}

async function summarize(question, answers) {
  const successful = answers.filter((a) => a.ok);
  if (!successful.length) {
    return "None of the models could answer this time — please try again in a moment.";
  }
  const digest = successful.map((a) => `${a.name}: ${a.text}`).join("\n");
  const system =
    "You write short, neutral summaries comparing several AI answers to the same question. " +
    "In 2-4 sentences: say where the answers broadly agree, name any real disagreement, and end with a clear plain-language takeaway. " +
    "Do not just repeat every answer; synthesize.";
  const prompt = `Question: ${question}\n\nAnswers:\n${digest}\n\nWrite the summary now.`;
  try {
    return await callGroq(system, prompt, { maxTokens: 200, temperature: 0.5 });
  } catch (err) {
    return "Summary unavailable right now, but you can compare the answers above directly.";
  }
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

    const answers = await askAllPersonas(question, selectedIds);
    const summary = await summarize(question, answers);

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
