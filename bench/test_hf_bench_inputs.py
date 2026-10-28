#!/usr/bin/env python3
"""make_inputs 的单测：确保 HF 基线与引擎侧的输入口径能对齐。

为什么值得测：`client_bench` 走 `/v1/chat/completions`，服务端会套 chat template；
如果 `hf_bench` 直接喂裸文本，"2048 token prompt"在两边就不是同一个输入，
由此得出的"比 transformers 快 N 倍"不成立。这段逻辑一旦退化，结论就会静默失真。
"""
from __future__ import annotations

import pytest

# hf_bench 需要 torch；CI 里为了速度不装 torch 时，这组测试应当**跳过**
# 而不是在收集阶段报 ImportError（那会让整份报告变红，掩盖真正的失败）。
# exc_type 必须显式写：pytest 9.1 起隐式捕获 ImportError 会变成错误。
torch = pytest.importorskip(
    "torch",
    reason="hf_bench 依赖 torch，CI 未安装时跳过",
    exc_type=ImportError,
)

from hf_bench import make_inputs  # noqa: E402


class FakeTokenizer:
    """只实现 make_inputs 用到的那两个入口。"""

    def __init__(self, template_len=12, raw_len=8, one_dim=False,
                 reject_kwarg=False):
        self.template_len = template_len
        self.raw_len = raw_len
        self.one_dim = one_dim
        self.reject_kwarg = reject_kwarg
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append({"kind": "template", **kw})
        if self.reject_kwarg and "enable_thinking" in kw:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        n = self.template_len
        ids = torch.arange(n).unsqueeze(0)          # [1, n]
        return ids.squeeze(0) if self.one_dim else ids

    def __call__(self, batch, **kw):
        self.calls.append({"kind": "raw", "n": len(batch)})
        return {"input_ids": torch.zeros(len(batch), self.raw_len, dtype=torch.long)}


def test_chat_template_batched_to_batch_size():
    tok = FakeTokenizer(template_len=12)
    out = make_inputs(tok, "hi", batch_size=3, chat_template=True)
    assert out["input_ids"].shape == (3, 12)
    assert out["attention_mask"].shape == (3, 12)
    assert all(v == 1 for v in out["attention_mask"].flatten().tolist())
    assert tok.calls[0]["kind"] == "template"
    assert tok.calls[0]["add_generation_prompt"] is True


def test_chat_template_disables_thinking():
    """与 client_bench 的 enable_thinking=False 保持一致，比纯解码。"""
    tok = FakeTokenizer()
    make_inputs(tok, "hi", 1, chat_template=True)
    assert tok.calls[0]["enable_thinking"] is False


def test_one_dim_template_output_is_promoted():
    tok = FakeTokenizer(template_len=5, one_dim=True)
    out = make_inputs(tok, "hi", 2, chat_template=True)
    assert out["input_ids"].shape == (2, 5)


def test_falls_back_when_tokenizer_rejects_enable_thinking():
    tok = FakeTokenizer(template_len=7, reject_kwarg=True)
    out = make_inputs(tok, "hi", 1, chat_template=True)
    assert out["input_ids"].shape == (1, 7)
    assert len(tok.calls) == 2                    # 第一次失败，第二次成功


def test_no_chat_template_uses_raw_path():
    tok = FakeTokenizer(raw_len=8)
    out = make_inputs(tok, "hi", 4, chat_template=False)
    assert out["input_ids"].shape == (4, 8)
    assert tok.calls[0]["kind"] == "raw"
    assert tok.calls[0]["n"] == 4


def test_prompt_tokens_differ_between_modes():
    """口径差别的量化：模板模式比裸文本多出模板自身的 token。"""
    tok = FakeTokenizer(template_len=12, raw_len=8)
    t = make_inputs(tok, "hi", 1, chat_template=True)["input_ids"].shape[1]
    r = make_inputs(tok, "hi", 1, chat_template=False)["input_ids"].shape[1]
    assert t > r, "模板模式的输入应当更长（含 system/user 标记）"
