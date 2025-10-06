// src/ml/session.ts
import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "/";
ort.env.wasm.numThreads = 1;   // evita multi-thread (precisa de cross-origin isolation)
ort.env.wasm.proxy = false;    // roda WASM no main thread (mais simples no dev)


export type SessionRefs = {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
  backend: "webgpu" | "wasm";
};

export async function createOnnxSession(modelUrl = "/models/gesture.onnx"): Promise<SessionRefs> {
  const canWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

  const tryProviders: Array<{ name: "webgpu" | "wasm"; opts?: ort.InferenceSession.SessionOptions }> = [
    { name: "webgpu", opts: { executionProviders: ["webgpu"] } as any },
    { name: "wasm",   opts: { executionProviders: ["wasm"], graphOptimizationLevel: "all" } as any },
  ];

  const order = canWebGPU ? tryProviders : tryProviders.slice(1);
  let lastErr: unknown = null;

  for (const prov of order) {
    try {
      const session = await ort.InferenceSession.create(modelUrl, prov.opts as any);
      const inputName  = session.inputNames?.[0]  ?? "input";
      const outputName = session.outputNames?.[0] ?? "output";
      return { session, inputName, outputName, backend: prov.name };
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error("Falha ao criar sessão ONNX. Detalhes: " + (lastErr as any)?.message);
}

export async function runOnce(
  refs: SessionRefs,
  data: Float32Array,
  shape: number[]
) {
  const tensor = new ort.Tensor("float32", data, shape);
  const feeds: Record<string, ort.Tensor> = { [refs.inputName]: tensor };
  const out = await refs.session.run(feeds);
  return out[refs.outputName];
}
