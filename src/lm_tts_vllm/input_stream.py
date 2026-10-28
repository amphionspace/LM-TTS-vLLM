"""Incremental text chunks for one native resumable LM-TTS request."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    token_ids: tuple[int, ...]
    final: bool
    audio_frames: int
    frames_before: int
    submitted_target_tokens: int


class TextStreamBuffer:
    """Commit token IDs once; wait for enough text rather than inventing padding.

    Each incoming text delta is tokenized independently. A caller with an LLM
    token stream can pass IDs directly. These committed boundaries are also
    the boundaries to use in a teacher-forced reference comparison.
    """
    def __init__(self, reference_ids, reference_frames, eos_id, max_frames, chunk_text=5, chunk_frames=10):
        if reference_frames < 0 or max_frames < 1 or min(chunk_text, chunk_frames) < 1:
            raise ValueError('Invalid text-stream dimensions')
        self.pending = list(reference_ids)
        self.reference_tokens = len(self.pending)
        self.reference_frames = reference_frames
        self.eos_id = eos_id
        self.max_frames = max_frames
        self.chunk_text, self.chunk_frames = chunk_text, chunk_frames
        self.index = self.sent = self.frames = 0
        self.finished = False
        self.exhausted = False

    def append(self, token_ids, *, final=False):
        if self.finished or self.exhausted:
            raise ValueError('Text input is already closed')
        self.pending.extend(token_ids)
        if final:
            self.pending.append(self.eos_id)
            self.finished = True

    def needed(self):
        """Text tokens still missing before the next segment can be submitted."""
        if self.finished or self.exhausted:
            return 0
        need = ((self.reference_frames // self.chunk_frames + 1) * self.chunk_text
                if self.index == 0 else self.chunk_text)
        return max(0, need - len(self.pending))

    def drain(self):
        while self.pending and not self.exhausted:
            need = ((self.reference_frames // self.chunk_frames + 1) * self.chunk_text
                    if self.index == 0 else self.chunk_text)
            if len(self.pending) < need and not self.finished:
                return
            size = min(need, len(self.pending))
            ids = tuple(self.pending[:size])
            del self.pending[:size]
            final = self.finished and not self.pending
            available = self.max_frames - self.frames
            frames = available if final else min(available,
                self.chunk_frames - self.reference_frames % self.chunk_frames if self.index == 0 else self.chunk_frames)
            self.sent += len(ids) - int(final)
            yield TextChunk(self.index, ids, final, frames, self.frames,
                            max(0, self.sent - self.reference_tokens))
            self.index += 1
            self.frames += frames
            if final or self.frames >= self.max_frames:
                self.exhausted = True


def chunk_plan(chunk_frames, final):
    from .interleaved import DecodePlan
    # The last pad is a segment stop, never a KV input or an audio frame.
    return DecodePlan((-1,) * chunk_frames + (0,), (final,) * chunk_frames + (False,), 0)
