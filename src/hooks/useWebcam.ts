// src/hooks/useWebcam.ts
import { useEffect, useRef, useState } from "react";

type WebcamOptions = {
  width?: number;
  height?: number;
  facingMode?: "user" | "environment";
};

export function useWebcam(opts: WebcamOptions = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const constraints: MediaStreamConstraints = {
      video: {
        width: opts.width ?? 640,
        height: opts.height ?? 480,
        facingMode: opts.facingMode ?? "user",
      },
      audio: false,
    };

    let stream: MediaStream | undefined;

    navigator.mediaDevices.getUserMedia(constraints)
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          return videoRef.current.play();
        }
      })
      .then(() => setReady(true))
      .catch((e) => setError(e?.message ?? String(e)));

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [opts.width, opts.height, opts.facingMode]);

  return { videoRef, ready, error };
}
