import { homedir } from "node:os";
import { join } from "node:path";

export const DECAY_CLASSES = [
  "permanent",
  "stable",
  "active",
  "session",
  "checkpoint",
] as const;
export type DecayClass = (typeof DECAY_CLASSES)[number];

/** TTL defaults in seconds per decay class. null = never expires. */
export const TTL_DEFAULTS: Record<DecayClass, number | null> = {
  permanent: null,
  stable: 90 * 24 * 3600, // 90 days
  active: 14 * 24 * 3600, // 14 days
  session: 24 * 3600, // 24 hours
  checkpoint: 4 * 3600, // 4 hours
};

export const MEMORY_CATEGORIES = [
  "preference",
  "fact",
  "decision",
  "entity",
  "other",
] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

const DEFAULT_OPENAI_MODEL = "text-embedding-3-small";
const DEFAULT_LOCAL_MODEL =
  "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf";
const DEFAULT_LANCE_PATH = join(homedir(), ".openclaw", "memory", "lancedb");
const DEFAULT_SQLITE_PATH = join(homedir(), ".openclaw", "memory", "facts.db");

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  [DEFAULT_LOCAL_MODEL]: 768,
  "embeddinggemma-300m-qat-Q8_0.gguf": 768,
};

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (dims) return dims;
  if (model.includes("embeddinggemma") || model.includes("300m")) return 768;
  throw new Error(`Unsupported embedding model: ${model}`);
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) throw new Error(`Environment variable ${envVar} is not set`);
    return envValue;
  });
}

export type HybridMemoryConfig = {
  embedding: {
    provider: "openai" | "local";
    model: string;
    apiKey?: string;
    baseUrl?: string;
    modelPath?: string;
    modelCacheDir?: string;
  };
  lanceDbPath: string;
  sqlitePath: string;
  autoCapture: boolean;
  autoRecall: boolean;
  captureMaxChars?: number;
};

export const hybridConfigSchema = {
  parse(value: unknown): HybridMemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory-hybrid config required");
    }
    const cfg = value as Record<string, unknown>;

    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding || typeof embedding !== "object") {
      throw new Error("embedding config required");
    }

    const provider =
      (embedding.provider as "openai" | "local") ?? "local";

    if (provider === "openai") {
      const apiKey = embedding.apiKey;
      if (typeof apiKey !== "string") {
        throw new Error("embedding.apiKey required when provider is openai");
      }
      const model =
        typeof embedding.model === "string" ? embedding.model : DEFAULT_OPENAI_MODEL;
      vectorDimsForModel(model);
      return {
        embedding: {
          provider: "openai",
          model,
          apiKey: resolveEnvVars(apiKey),
          baseUrl: typeof embedding.baseUrl === "string" ? embedding.baseUrl : undefined,
        },
        lanceDbPath:
          typeof cfg.lanceDbPath === "string" ? cfg.lanceDbPath : DEFAULT_LANCE_PATH,
        sqlitePath:
          typeof cfg.sqlitePath === "string" ? cfg.sqlitePath : DEFAULT_SQLITE_PATH,
        autoCapture: cfg.autoCapture !== false,
        autoRecall: cfg.autoRecall !== false,
        captureMaxChars:
          typeof cfg.captureMaxChars === "number" ? cfg.captureMaxChars : 500,
      };
    }

    const modelPath =
      typeof embedding.modelPath === "string"
        ? embedding.modelPath
        : typeof embedding.model === "string"
          ? embedding.model
          : DEFAULT_LOCAL_MODEL;
    vectorDimsForModel(modelPath);

    return {
      embedding: {
        provider: "local",
        model: modelPath,
        modelPath,
        modelCacheDir:
          typeof embedding.modelCacheDir === "string"
            ? embedding.modelCacheDir
            : undefined,
      },
      lanceDbPath:
        typeof cfg.lanceDbPath === "string" ? cfg.lanceDbPath : DEFAULT_LANCE_PATH,
      sqlitePath:
        typeof cfg.sqlitePath === "string" ? cfg.sqlitePath : DEFAULT_SQLITE_PATH,
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
      captureMaxChars:
        typeof cfg.captureMaxChars === "number" ? cfg.captureMaxChars : 500,
    };
  },
};
