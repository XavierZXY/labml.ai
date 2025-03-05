from typing import List

import torch
from labml import lab, monit
from transformers import BertModel, BertTokenizer


class BERTChunkEmbeddings:
    def __init__(self, device: torch.device):
        self.device = device
        with monit.section("Loading BERT"):
            self.model = BertModel.from_pretrained("bert-base-uncased").to(
                device
            )
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def _trim_chunk(chunk: str):
        stripped = chunk.strip()
        parts = stripped.split()
        # 移除第一块和最后一块碎片
        stripped = stripped[len(parts[0]) : -len(parts[-1])]
        # 移除多余的空格
        stripped = stripped.strip()
        if not stripped:
            return chunk

        return stripped

    def __call__(self, chunks: List[str]):
        with torch.no_grad():
            trimmed_chunks = [self._trim_chunk(chunk) for chunk in chunks]
            # 分词
            tokens = self.tokenizer(
                trimmed_chunks,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=512,
            )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        token_type_ids = tokens["token_type_ids"].to(self.device)
        # 获取BERT模型的输出
        output = self.model(input_ids, attention_mask, token_type_ids)
        # 返回最后一层的输出
        state = output["last_hidden_state"]
        emb = (state * attention_mask[:, :, None]).sum(dim=1) / attention_mask[
            :, :, None
        ].sum(dim=1)
        return emb


def _test():
    from labml.logger import inspect

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BERTChunkEmbeddings(device)
    text = ["This is a test", "This is another test"]
    encoding_input = bert.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    inspect(encoding_input, _expand=True)
    output = bert.model(
        input_ids=encoding_input["input_ids"].to(device),
        attention_mask=encoding_input["attention_mask"].to(device),
    )
    inspect(output["last_hidden_state"].shape, _expand=True)
    inspect(bert(text))


if __name__ == "__main__":
    _test()
