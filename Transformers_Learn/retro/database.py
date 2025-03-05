from typing import List, Optional

import faiss
import numpy as np
import torch
from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings


def build_database(
    chunk_len: int = 16,
    batch_size: int = 64,
    d_emb: int = 768,
    n_centeroids: int = 256,
    code_size: int = 64,
    n_probe: int = 8,
    n_train: int = 50_000,
):
    """_summary_

    Args:
        chunk_len (int, optional): the length of a chunk. Defaults to 16.
        batch_size (int, optional): batch size. Defaults to 64.
        d_emb (int, optional): the number of features in Bert(N) embeddings. Defaults to 768.
        n_code_size (int, optional): the number of lists in the index. Defaults to 64.
        n_probe (int, optional): the number fo lists in the index. Defaults to 8.
        n_train (int, optional): _description_. Defaults to 50_000.
    """
    dataset = TextFileDataset(
        lab.get_data_path() / "tiny_shakespeare.txt",
        list,
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    )
    text = dataset.train
    # split the text into chunks
    chunks = [
        text[i : i + chunk_len]
        for i in range(0, len(text), chunk_len)
        if i + chunk_len * 2 < len(text)
    ]
    # get the offset of the chunks
    chunk_offsets = np.array(
        [
            i
            for i in range(0, len(text), chunk_len)
            if i + chunk_len * 2 < len(text)
        ]
    )
    n_chunks = len(chunks)
    # 初始化BERT以获取BERT(N)嵌入
    bert = BERTChunkEmbeddings(torch.device("cuda"))
    # 通过处理每次迭代的块来获取区块嵌入
    chunk_emb = []
    for i in monit.iterate(
        "Getting chunk embeddings", range(0, n_chunks, batch_size)
    ):
        chunk_emb.append(bert(chunks[i : i + batch_size])).cpu()
    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()

    # 创建faiss
    quantizer = faiss.IndexFlatL2(d_emb)
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centeroids, code_size, 8)
    index.nprobe = n_probe
    # 获取区块索引的随机样本
    random_sample = np.random.choice(
        np.arange(n_chunks), size=[min(n_train, n_chunks)], replace=False
    )
    # 训练索引
    with monit.section("Training index"):
        index.train(chunk_emb[random_sample])

    # 添加区块嵌入到索引
    for s in monit.iterate("Adding embeddings", range(0, n_chunks, 1024)):
        e = min(s + 1024, n_chunks)
        index.add_with_ids(chunk_emb[s:e], chunk_offsets[s:e])
    # 保存索引
    with monit.section("Saving index"):
        faiss.write_index(index, str(lab.get_data_path() / "retro.index"))


class RetroIndex:
    def __init__(
        self,
        chunk_len: int = 16,
        n_probe: int = 8,
        n_neighbors: int = 2,
        n_extra: int = 2,
        exclude_neighbor_span: int = 8,
    ):
        """检索最近邻居

        Args:
            chunk_len (int, optional): _description_. Defaults to 16.
            n_probe (int, optional): 要探测的列表的数量. Defaults to 8.
            n_neighbors (int, optional): 要得到的检索的邻居的数量. Defaults to 2.
            n_extra (int, optional): 要检索的额外的邻居的数量. Defaults to 2.
            exclude_neighbor_span (int, optional): 检查重叠时要避免的额外的文本长度. Defaults to 8.
        """
        self.chunk_len = chunk_len
        self.n_neighbors = n_neighbors
        self.n_extra = n_extra
        self.exclude_neighbor_span = exclude_neighbor_span

        self.bert = BERTChunkEmbeddings(torch.device("cuda"))
        with monit.section("Loading index"):
            self.index = faiss.read_index(
                str(lab.get_data_path() / "retro.index")
            )
            self.index.nprobe = n_probe

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        """筛选与查询重叠的邻居

        Args:
            offset (int): _description_
            neighbor_offsets (List[int]): _description_
        """
        return [
            n
            for n in neighbor_offsets
            if n < offset - (self.chunk_len + self.exclude_neighbor_span)
            or n > offset + (self.chunk_len + self.exclude_neighbor_span)
        ]

    def __call__(self, query_chunks: List[str], offsets: Optional[List[int]]):
        emb = self.bert(query_chunks).cpu()
        distance, neighbors_offsets = self.index.search(
            emb.numpy(), self.n_neighbors + self.n_extra
        )
        if offsets is None:
            neighbors_offsets = [
                self.filter_neighbors(off, n_off)
                for off, n_off in zip(offsets, neighbors_offsets)
            ]
        neighbors_offsets = [
            n_off[: self.n_neighbors] for n_off in neighbors_offsets
        ]

        return neighbors_offsets


if __name__ == "__main__":
    build_database()
