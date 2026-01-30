import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type EmbeddingProvider = "openai" | "local";

export type MemoryConfig = {
  embedding: {
    provider: EmbeddingProvider;
    model?: string;
    apiKey?: string;
    // Local embedding options
    local?: {
      modelPath?: string;
      modelCacheDir?: string;
    };
  };
  dbPath?: string;
  autoCapture?: boolean;
  autoRecall?: boolean;
};

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

const DEFAULT_OPENAI_MODEL = "text-embedding-3-small";
const DEFAULT_LOCAL_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const LEGACY_STATE_DIRS: string[] = [];

function resolveDefaultDbPath(): string {
  const home = homedir();
  const preferred = join(home, ".openclaw", "memory", "lancedb");
  try {
    if (fs.existsSync(preferred)) {
      return preferred;
    }
  } catch {
    // best-effort
  }

  for (const legacy of LEGACY_STATE_DIRS) {
    const candidate = join(home, legacy, "memory", "lancedb");
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // best-effort
    }
  }

  return preferred;
}

const DEFAULT_DB_PATH = resolveDefaultDbPath();

// OpenAI embedding dimensions
const OPENAI_EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

// Default dimension for local models (embeddinggemma-300M outputs 768-dim vectors)
const DEFAULT_LOCAL_EMBEDDING_DIM = 768;

function assertAllowedKeys(
  value: Record<string, unknown>,
  allowed: string[],
  label: string,
) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

export function vectorDimsForModel(model: string, provider: EmbeddingProvider): number {
  if (provider === "local") {
    // Local models have varying dimensions; default to embeddinggemma's 768
    // TODO: Could detect from model metadata in the future
    return DEFAULT_LOCAL_EMBEDDING_DIM;
  }

  const dims = OPENAI_EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(`Unsupported OpenAI embedding model: ${model}. Supported: ${Object.keys(OPENAI_EMBEDDING_DIMENSIONS).join(", ")}`);
  }
  return dims;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function resolveEmbeddingModel(embedding: Record<string, unknown>, provider: EmbeddingProvider): string {
  const model = typeof embedding.model === "string" ? embedding.model : undefined;

  if (provider === "local") {
    return model || DEFAULT_LOCAL_MODEL;
  }

  // OpenAI provider
  const resolvedModel = model || DEFAULT_OPENAI_MODEL;
  vectorDimsForModel(resolvedModel, provider); // Validate
  return resolvedModel;
}

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(cfg, ["embedding", "dbPath", "autoCapture", "autoRecall"], "memory config");

    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding) {
      throw new Error("embedding config is required");
    }
    assertAllowedKeys(embedding, ["provider", "apiKey", "model", "local"], "embedding config");

    // Determine provider (default to "openai" for backwards compatibility)
    const provider: EmbeddingProvider = embedding.provider === "local" ? "local" : "openai";

    // Validate apiKey requirement based on provider
    if (provider === "openai" && typeof embedding.apiKey !== "string") {
      throw new Error("embedding.apiKey is required when using OpenAI provider");
    }

    const model = resolveEmbeddingModel(embedding, provider);

    // Parse local config if present
    let localConfig: MemoryConfig["embedding"]["local"] | undefined;
    if (embedding.local && typeof embedding.local === "object") {
      const local = embedding.local as Record<string, unknown>;
      assertAllowedKeys(local, ["modelPath", "modelCacheDir"], "embedding.local config");
      localConfig = {
        modelPath: typeof local.modelPath === "string" ? local.modelPath : undefined,
        modelCacheDir: typeof local.modelCacheDir === "string" ? local.modelCacheDir : undefined,
      };
    }

    return {
      embedding: {
        provider,
        model,
        apiKey: typeof embedding.apiKey === "string" ? resolveEnvVars(embedding.apiKey) : undefined,
        local: localConfig,
      },
      dbPath: typeof cfg.dbPath === "string" ? cfg.dbPath : DEFAULT_DB_PATH,
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
    };
  },
  uiHints: {
    "embedding.provider": {
      label: "Embedding Provider",
      help: "Choose 'openai' for remote embeddings or 'local' for on-device embeddings using node-llama-cpp",
      options: ["openai", "local"],
    },
    "embedding.apiKey": {
      label: "OpenAI API Key",
      sensitive: true,
      placeholder: "sk-proj-...",
      help: "API key for OpenAI embeddings (required if provider is 'openai', or use ${OPENAI_API_KEY})",
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_OPENAI_MODEL,
      help: "Model to use for embeddings. For OpenAI: text-embedding-3-small/large. For local: HuggingFace GGUF path.",
    },
    "embedding.local.modelPath": {
      label: "Local Model Path",
      placeholder: DEFAULT_LOCAL_MODEL,
      help: "Path to local GGUF embedding model (e.g., hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf)",
      advanced: true,
    },
    "embedding.local.modelCacheDir": {
      label: "Model Cache Directory",
      placeholder: "~/.cache/node-llama-cpp",
      help: "Directory to cache downloaded models",
      advanced: true,
    },
    dbPath: {
      label: "Database Path",
      placeholder: "~/.openclaw/memory/lancedb",
      advanced: true,
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information from conversations",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
  },
};
