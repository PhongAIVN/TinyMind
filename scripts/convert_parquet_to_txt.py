from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm


# Xác định thư mục gốc của project dựa vào vị trí của script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "dataset" / "PRETRAIN" / "eval"

print(f"Parquet dir: {PARQUET_DIR}")

parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
print(f"Vị trí ({len(parquet_files)} files): {[f.name for f in parquet_files]}")

if not parquet_files:
    raise FileNotFoundError(f"Không tìm thấy file parquet nào trong thư mục: {PARQUET_DIR}")


for idx, parquet_file in enumerate(parquet_files, start=1):
    output_file = PARQUET_DIR / f"train_vi_{idx:03d}.txt"

    print(f"\nConverting {parquet_file.name} -> {output_file.name}")

    # Chỉ đọc cột 'text' để tiết kiệm RAM và tăng tốc (bỏ qua cột embeddings nặng)
    table = pq.read_table(parquet_file, columns=["text"])

    texts = table["text"].to_pylist()

    count = 0

    with output_file.open("w", encoding="utf-8") as out:
        for text in tqdm(texts, desc=output_file.name):
            if text is None:
                continue

            text = text.strip()

            if len(text) < 100:
                continue

            out.write(text)
            out.write("\n\n")
            count += 1

    print(f"Saved {count:,} documents")


print("\nDONE")