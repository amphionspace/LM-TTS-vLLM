"""Keep the demo's Transformers runtime separate from vLLM's runtime."""
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess


class TalkerWorker:
    def __init__(self, python, model, log, precision="bf16", timeout=900, engine_mode="reference"):
        left, right = socket.socketpair()
        self.connection = Connection(left.detach())
        self.timeout = timeout
        self.steps = []
        self.log = open(log, "w")
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["LM_TTS_PRECISION"] = precision
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_PLUGINS"] = "lm_tts"
        env["TOKENIZERS_PARALLELISM"] = "false"
        try:
            self.process = subprocess.Popen(
                [str(python), "-m", "lm_tts_vllm.worker", "--fd", str(right.fileno()), "--model", str(model),
                 "--engine-mode", engine_mode],
                pass_fds=(right.fileno(),), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        finally:
            right.close()
        try:
            self.info = self.receive()
        except BaseException:
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"vLLM worker timed out; inspect {self.log.name}")
        try:
            value = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"vLLM worker exited; inspect {self.log.name}") from exc
        if "error" in value:
            raise RuntimeError(value["error"])
        return value

    def hidden(self, embeddings):
        import torch
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Parity frontend supports a single unpadded utterance")
        self.connection.send(embeddings[0].detach().float().cpu().numpy())
        response = self.receive()
        result = response.pop("hidden")
        self.steps.append(response)
        return torch.from_numpy(result).to(embeddings).reshape(1, 1, -1)

    def reset_cache(self):
        self.connection.send({"reset_cache": True})
        if not self.receive()["reset"]:
            raise RuntimeError("Could not reset the prefix cache before free-running measurement")

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                try:
                    self.connection.send(None)
                    self.process.wait(timeout=15)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    self.process.wait(timeout=15)
        self.connection.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
