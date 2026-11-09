"""SFT: nhãn chỉ phủ assistant, khuôn train = khuôn sinh chữ, init_from nạp đúng trọng số."""

import json
from pathlib import Path

import pytest
import torch

from core.config import ModelConfig
from scripts.build_sft_data import build_sft_data, viquad_pair
from trainer.losses import IGNORE_INDEX
from trainer.sft_dataset import ChatEncoder, SFTDataset
from trainer.train_pretrain import run_training, train

REAL_TOKENIZER = Path("tokenizer/tokenizer.json")
pytestmark = pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="cần tokenizer thật")

CONVERSATION = [
    {"role": "system", "content": "Bạn là trợ lý."},
    {"role": "user", "content": "1 + 1 bằng mấy?"},
    {"role": "assistant", "content": "Bằng 2."},
    {"role": "user", "content": "Còn 2 + 2?"},
    {"role": "assistant", "content": "Bằng 4."},
]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(REAL_TOKENIZER))


def write_jsonl(path: Path, conversations: list) -> Path:
    path.write_text("".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in conversations), encoding="utf-8")
    return path


def test_labels_cover_only_assistant_replies(tokenizer):
    encoder = ChatEncoder(tokenizer)
    ids, labels = encoder.encode_conversation(CONVERSATION)

    trained = [token for token, label in zip(ids, labels) if label != IGNORE_INDEX]
    assert trained == [label for label in labels if label != IGNORE_INDEX]
    expected = []
    for reply in ("Bằng 2.", "Bằng 4."):
        expected += tokenizer.encode(reply, add_special_tokens=False).ids + [encoder.im_end_id]
    assert trained == expected


def test_prompt_matches_training_prefix(tokenizer):
    """Lúc sinh chữ, prompt phải trùng từng token với phần trước câu trả lời lúc train."""
    encoder = ChatEncoder(tokenizer)
    messages = [{"role": "user", "content": "Thủ đô Việt Nam?"}, {"role": "assistant", "content": "Hà Nội."}]
    ids, labels = encoder.encode_conversation(messages)
    prompt = encoder.encode_prompt("Thủ đô Việt Nam?")

    assert ids[: len(prompt)] == prompt
    assert labels[len(prompt)] != IGNORE_INDEX  # token đầu tiên của câu trả lời là token được học


def test_unknown_role_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="role"):
        ChatEncoder(tokenizer).encode_conversation([{"role": "bot", "content": "x"}])


def test_dataset_shifts_targets_and_pads_with_ignore(tokenizer, tmp_path):
    path = write_jsonl(tmp_path / "sft.jsonl", [CONVERSATION])
    dataset = SFTDataset(path, tokenizer, seq_len=64)
    x, y = dataset[0]
    ids, labels = ChatEncoder(tokenizer).encode_conversation(CONVERSATION)

    assert x.shape == y.shape == (64,)
    assert x[: len(ids)].tolist() == ids[:64]
    assert y[: len(labels) - 1].tolist() == labels[1:]
    assert (y[len(labels) - 1 :] == IGNORE_INDEX).all()


def test_dataset_drops_conversation_truncated_before_reply(tokenizer, tmp_path):
    long_question = [{"role": "user", "content": "chữ " * 200}, {"role": "assistant", "content": "ok"}]
    path = write_jsonl(tmp_path / "sft.jsonl", [long_question, CONVERSATION])
    dataset = SFTDataset(path, tokenizer, seq_len=64)

    assert len(dataset) == 1 and dataset.dropped == 1


def test_viquad_pair_puts_context_in_question():
    record = {"context": "Đoạn văn.", "question": "Hỏi?", "answers": {"text": ["Đáp"]}, "is_impossible": False}
    assert viquad_pair(record) == ("Đoạn văn.\n\nHỏi?", "Đáp")
    assert viquad_pair({**record, "is_impossible": True}) is None


def test_build_sft_data_splits_and_merges_extra(tmp_path):
    raw = tmp_path / "raw" / "vi_alpaca"
    raw.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"instruction": f"câu {i}", "input": "", "output": f"đáp {i}"} for i in range(450)]
    pq.write_table(pa.Table.from_pylist(rows), raw / "data.parquet")
    extra = write_jsonl(tmp_path / "gold.jsonl", [CONVERSATION])

    counts = build_sft_data(tmp_path / "raw", tmp_path / "out", ["vi_alpaca"], max_per_source=400, extra_files=[extra])

    assert counts == {"train": 399, "eval": 2}  # 400 alpaca (2 vào eval) + 1 hội thoại extra
    first = json.loads((tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["messages"] == [{"role": "user", "content": "câu 0"}, {"role": "assistant", "content": "đáp 0"}]


def test_init_from_loads_pretrained_weights(tokenizer, tmp_path, make_train_config):
    """LR gần 0: trọng số sau SFT phải gần như bằng trọng số pretrain nạp vào."""
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=16, n_layers=1, n_heads=2,
        ffn_hidden=32, max_seq_len=64, rope_theta=10000.0, norm_eps=1e-6, tie_embedding=True,
    )
    pretrain_dir = tmp_path / "pretrain"
    train(model_config, make_train_config(out_dir=str(pretrain_dir), max_steps=1))
    pretrain_path = pretrain_dir / "pretrain.pt"

    data = write_jsonl(tmp_path / "sft.jsonl", [CONVERSATION] * 8)
    dataset = SFTDataset(data, tokenizer, seq_len=64)
    sft_config = make_train_config(
        out_dir=str(tmp_path / "sft"), init_from=str(pretrain_path), lr=1e-12, min_lr=0.0, seq_len=64, max_steps=1
    )
    stats = run_training(model_config, sft_config, dataset, dataset, "sft.pt", resume=False)

    assert torch.isfinite(torch.tensor(stats["train_loss"]))
    before = torch.load(pretrain_path, weights_only=False)["model"]
    after = torch.load(tmp_path / "sft" / "sft.pt", weights_only=False)["model"]
    for name in before:
        torch.testing.assert_close(after[name], before[name], atol=1e-6, rtol=0)


def test_missing_init_from_file_raises(tokenizer, tmp_path, make_train_config, tiny_model_config):
    data = write_jsonl(tmp_path / "sft.jsonl", [CONVERSATION])
    config = make_train_config(init_from=str(tmp_path / "khong_co.pt"))
    with pytest.raises(FileNotFoundError, match="init_from"):
        run_training(tiny_model_config, config, SFTDataset(data, tokenizer, 64), SFTDataset(data, tokenizer, 64), "sft.pt")


def test_sharegpt_multi_turn_maps_roles():
    from scripts.build_sft_data import sharegpt_messages

    record = {"conversations": [
        {"from": "human", "value": "Chào"}, {"from": "gpt", "value": "Chào bạn"},
        {"from": "human", "value": "Khỏe không?"}, {"from": "gpt", "value": "Khỏe"},
    ]}
    assert [m["role"] for m in sharegpt_messages(record)] == ["user", "assistant", "user", "assistant"]
    assert sharegpt_messages({"conversations": [{"from": "bot", "value": "x"}]}) is None
    assert sharegpt_messages({"conversations": [{"from": "human", "value": "chỉ hỏi"}]}) is None


def test_build_sft_data_reads_ultrachat_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = tmp_path / "raw" / "ultrachat_vi" / "data"
    folder.mkdir(parents=True)
    messages = [{"role": "user", "content": "Hỏi"}, {"role": "assistant", "content": "Đáp"}]
    pq.write_table(pa.Table.from_pylist([{"prompt": "Hỏi", "messages": messages}] * 3), folder / "train.parquet")

    counts = build_sft_data(tmp_path / "raw", tmp_path / "out", ["ultrachat_vi"], max_per_source=10)

    assert counts == {"train": 3, "eval": 0}
    first = json.loads((tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["messages"] == messages
