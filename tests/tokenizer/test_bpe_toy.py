import pytest

from tokenizer.bpe_toy import ByteBPE, count_pairs, merge

TRAIN_TEXT = "Việt Nam là một quốc gia. Người Việt nói tiếng Việt. Hoàng Sa và Trường Sa là của Việt Nam " * 20


def test_count_pairs():
    assert count_pairs([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge_replaces_every_occurrence():
    assert merge([1, 2, 3, 1, 2], pair=(1, 2), new_id=99) == [99, 3, 99]


def test_merge_does_not_overlap():
    assert merge([7, 7, 7], pair=(7, 7), new_id=99) == [99, 7]


def test_untrained_encode_is_utf8_bytes():
    bpe = ByteBPE()

    assert bpe.encode("ệ") == [0xE1, 0xBB, 0x87]  # "ệ" = 3 byte UTF-8


def test_vocab_size_below_256_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        ByteBPE().train(TRAIN_TEXT, vocab_size=100)


def test_training_creates_requested_vocab_size():
    bpe = ByteBPE()
    bpe.train(TRAIN_TEXT, vocab_size=300)

    assert len(bpe.vocab) == 300
    assert len(bpe.merges) == 300 - 256


def test_training_twice_starts_fresh():
    """Train lần 2 phải bỏ hết kết quả lần 1, không trộn luật cũ với vocab mới."""
    bpe = ByteBPE()
    bpe.train("Việt Nam " * 20, vocab_size=270)
    bpe.train("xin chào " * 20, vocab_size=270)
    text = "Việt Nam"

    assert len(bpe.merges) == 270 - 256
    assert bpe.decode(bpe.encode(text)) == text


def test_training_makes_sequences_shorter():
    bpe = ByteBPE()
    bpe.train(TRAIN_TEXT, vocab_size=300)
    text = "Người Việt nói tiếng Việt."

    assert len(bpe.encode(text)) < len(text.encode("utf-8"))


def test_repeated_character_becomes_one_token():
    bpe = ByteBPE()
    bpe.train("ệ" * 50, vocab_size=258)  # 2 lần gộp: (E1,BB) rồi (256,87)

    assert len(bpe.encode("ệ")) == 1


@pytest.mark.parametrize("text", ["Người Việt nói tiếng Việt.", "Chưa từng thấy: ỹ ữ 😀", ""])
def test_decode_encode_round_trip(text):
    bpe = ByteBPE()
    bpe.train(TRAIN_TEXT, vocab_size=300)

    assert bpe.decode(bpe.encode(text)) == text