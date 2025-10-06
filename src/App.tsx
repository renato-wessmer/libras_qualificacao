// src/App.tsx
import { useMemo, useState } from "react";
import { useWebcam } from "./hooks/useWebcam";
import { createOnnxSession, runOnce } from "./ml/session";
import type { SessionRefs } from "./ml/session";

export default function App() {
  const { videoRef, ready, error } = useWebcam({ width: 640, height: 480 });
  const [session, setSession] = useState<SessionRefs | null>(null);
  const [msg, setMsg] = useState<string>("");

  const hasWebGPU = useMemo(
    () => typeof navigator !== "undefined" && "gpu" in navigator,
    []
  );

  async function handleLoadModel() {
    setMsg("Carregando modelo...");
    try {
      // quando criarmos a pasta, o modelo ficará em /public/models/gesture.onnx
      const s = await createOnnxSession("/models/gesture.onnx");
      setSession(s);
      setMsg(`Modelo carregado com ${s.backend.toUpperCase()}.`);
    } catch (e: any) {
      setMsg("Erro ao carregar modelo: " + (e?.message ?? String(e)));
    }
  }

  async function handleTestInference() {
    if (!session) return setMsg("Carregue o modelo antes.");
    const SEQ_LEN = 30;
    const FEATURE_DIM = 150; // ajuste para o seu modelo real
    const dummy = new Float32Array(SEQ_LEN * FEATURE_DIM);
    const output = await runOnce(session, dummy, [1, SEQ_LEN, FEATURE_DIM]);
    const arr = (output as any).data as Float32Array | number[];
    setMsg(
      `Inferência OK. Saída[0..4]= ${Array.from(arr)
        .slice(0, 5)
        .map((n) => Number(n).toFixed(4))
        .join(", ")}`
    );
  }

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 16 }}>
      <div style={{ width: 920, maxWidth: "100%", display: "grid", gap: 16 }}>
        <header style={{ textAlign: "center" }}>
          <h1 style={{ margin: 0 }}>LIBRA QUALIFICAÇÃO</h1>
          <p style={{ margin: 4, opacity: 0.8 }}>
            React + ONNX Runtime (WebGPU/WASM).{" "}
            {hasWebGPU ? "WebGPU disponível." : "WebGPU indisponível — usando WASM."}
          </p>
        </header>

        <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
            <h3 style={{ marginTop: 0 }}>Câmera</h3>
            {error && <p style={{ color: "crimson" }}>Erro: {error}</p>}
            <video
              ref={videoRef}
              playsInline
              muted
              style={{ width: "100%", borderRadius: 8, background: "#000" }}
            />
            <p style={{ fontSize: 12, opacity: 0.7, marginTop: 8 }}>
              {ready ? "Webcam pronta." : "Solicitando permissão da câmera..."}
            </p>
          </div>

          <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
            <h3 style={{ marginTop: 0 }}>Modelo</h3>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={handleLoadModel} style={btnStyle}>Carregar modelo</button>
              <button onClick={handleTestInference} style={btnStyle} disabled={!session}>
                Rodar inferência (dummy)
              </button>
            </div>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 13, background: "#f8fafc", padding: 8, borderRadius: 8, marginTop: 12 }}>
{msg || "Aguardando..."}
            </pre>
          </div>
        </section>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: 10,
  border: "1px solid #e5e7eb",
  background: "white",
  cursor: "pointer",
};
