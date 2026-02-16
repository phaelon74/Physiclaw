FROM node:22-bookworm

# Install Bun (required for build scripts)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

RUN corepack enable

# Build tools for native modules (better-sqlite3, node-llama-cpp)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app

ARG PHYSICLAW_DOCKER_APT_PACKAGES=""
RUN if [ -n "$PHYSICLAW_DOCKER_APT_PACKAGES" ]; then \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $PHYSICLAW_DOCKER_APT_PACKAGES && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
    fi

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY patches ./patches
COPY scripts ./scripts
COPY extensions ./extensions

RUN pnpm install

COPY . .
RUN pnpm build

ENV NODE_ENV=production

# Security hardening: Run as non-root user
RUN chown -R node:node /app
USER node

# Zero telemetry by default
ENV PHYSICLAW_TELEMETRY=off
ENV PHYSICLAW_PHONE_HOME=disabled

# Start gateway server. Binds to loopback (127.0.0.1) by default for security.
# For container networking, override with --bind lan
CMD ["node", "physiclaw.mjs", "gateway", "--allow-unconfigured"]
