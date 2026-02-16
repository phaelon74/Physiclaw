import { ImageResponse } from "next/og";

export const alt = "Physiclaw — Functional AI Agents on Your Hardware";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#001427",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "24px",
            padding: "80px",
          }}
        >
          <div
            style={{
              fontSize: 72,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              lineHeight: 1.1,
              textAlign: "center",
              display: "flex",
              flexWrap: "wrap",
              justifyContent: "center",
              alignItems: "baseline",
              gap: "8px",
            }}
          >
            <span style={{ color: "#f7e2aa" }}>Functional AI agents on </span>
            <span style={{ color: "#F4D58D" }}>your hardware</span>
            <span style={{ color: "#f7e2aa" }}>.</span>
          </div>
          <div
            style={{
              fontSize: 28,
              color: "#8aa89b",
              textAlign: "center",
              maxWidth: "800px",
            }}
          >
            Open-source software that runs AI agents entirely on your own servers. No cloud, no telemetry, no lock-in.
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              marginTop: "16px",
            }}
          >
            <span
              style={{
                fontSize: 18,
                color: "#4a5f55",
                fontFamily: "monospace",
              }}
            >
              Apache 2.0 · Self-hosted · Air-gap ready
            </span>
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}
