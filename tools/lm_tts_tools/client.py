"""Keep the demo's Transformers runtime separate from vLLM's runtime."""
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess


class TalkerWorker:
    def __init__(self, python, model, log, precision="bf16", timeout=900, engine_mode="reference",
                 attention_backend="FLASH_ATTN", code_predictor_engine="pytorch"):
        left, right = socket.socketpair()
        self.connection = Connection(left.detach())
        self.timeout = timeout
        self.steps = []
        self.predictor_steps = []
        self.log = open(log, "w")
        env = os.environ.copy()
        tool_root = Path(__file__).resolve().parents[1]
        paths = [str(tool_root)]
        plugin_src = tool_root.parent / "src"
        if (plugin_src / "lm_tts_vllm").is_dir():
            paths.append(str(plugin_src))
        env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
        env["LM_TTS_PRECISION"] = precision
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_PLUGINS"] = "lm_tts"
        env["TOKENIZERS_PARALLELISM"] = "false"
        try:
            self.process = subprocess.Popen(
                [str(python), "-m", "lm_tts_tools.worker", "--fd", str(right.fileno()), "--model", str(model),
                 "--engine-mode", engine_mode, "--attention-backend", attention_backend,
                 "--code-predictor-engine", code_predictor_engine],
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

    def predict_codes(self, inputs):
        import torch
        self.connection.send({"predict_codes": inputs.detach().float().cpu().numpy()})
        response = self.receive()
        codes = response.pop("codes")
        self.predictor_steps.append(response)
        return torch.from_numpy(codes).to(device=inputs.device, dtype=torch.long)

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
