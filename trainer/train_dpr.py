from pathlib import Path
import sys

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import os
import re
import time
from typing import List, Literal, Union
import warnings

import datasets
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    BertModel,
    BertTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
import wandb

from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig
from sentence_transformers import SentenceTransformer
from trainer.trainer_utils import (
    Logger,
    MiniMindForCausalLM,
    SkipBatchSampler,
    init_distributed_mode,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


# ----- dataset utils -----


def save_jsonl(lines: Union[list[dict], pd.DataFrame], path, overwrite=False):
    if isinstance(lines, pd.DataFrame):
        lines = lines.to_dict(orient="records")

    path = Path(path)
    if not overwrite:
        assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, line in enumerate(lines):
            json_line = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
            if i == 0:
                f.write(json_line)
            else:
                f.write("\n" + json_line)
    print(f"Saved {len(lines)} lines to {path.as_posix()}")


def load_jsonl(path):
    data = pd.read_json(path_or_buf=path, lines=True)
    return data


def load_dataset(
    dataset_name="ChineseSquad",
    max_query_len=64,
    max_passage_len=128,
    split=True,
):
    if dataset_name == "ChineseSquad":
        dataset_name = "lighteval/ChineseSquad"
    elif dataset_name == "natural-questions":
        dataset_name = "sentence-transformers/natural-questions"
    else:
        raise NotImplementedError

    dataset = datasets.load_dataset(dataset_name)
    if dataset_name == "sentence-transformers/natural-questions":
        dataset = dataset.filter(
            lambda x: len(x["query"]) <= max_query_len
            and len(x["answer"]) <= max_passage_len
        )
        if split:
            dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    else:
        dataset = dataset.filter(
            lambda x: len(x["question"]) <= max_query_len
            and len(x["context"]) <= max_passage_len
        )
    return dataset


class DPRDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_query_len=64,
        max_passage_len: int = 128,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.dataset = dataset
        if "query" in self.dataset.features:
            self.dataset_name = "nq"
        else:
            self.dataset_name = "cs"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        def encode(key, max_length):
            if isinstance(self.tokenizer, BertTokenizerFast):
                text = sample[key]
            else:
                text = self.tokenizer.bos_token + sample[key] + self.tokenizer.eos_token
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                padding_side="right",
                padding="max_length",
                return_token_type_ids=False,
            )
            inputs["input_ids"] = inputs["input_ids"].view(-1)
            inputs["attention_mask"] = inputs["attention_mask"].view(-1)
            return inputs

        if self.dataset_name == "cs":  # ChineseSquad
            query_inputs = encode("question", self.max_query_len)
            passage_inputs = encode("context", self.max_passage_len)
        else:  # natural-questions
            query_inputs = encode("query", self.max_query_len)
            passage_inputs = encode("answer", self.max_passage_len)
        return query_inputs, passage_inputs

    def tolist(self):
        if self.dataset_name == "cs":  # ChineseSquad
            queries = list(self.dataset["question"])
            passages = list(self.dataset["context"])
        else:
            queries = list(self.dataset["query"])
            passages = list(self.dataset["answer"])
        return queries, passages


# ----- retriever -----


class Encoder(nn.Module):
    """
    Adapted from CMU-11-667 (https://cmu-llms.org/) EncoderModel
    """

    def __init__(self, encoder: PreTrainedModel, temperature: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(
        self,
        query_inputs: BatchEncoding = None,
        passage_inputs: BatchEncoding = None,
    ):
        q_embeddings = (
            self.encode(query_inputs) if query_inputs else None
        )  # (n_queries, hidden_dim)
        p_embeddings = (
            self.encode(passage_inputs) if passage_inputs else None
        )  # (n_passages, hidden_dim)

        # for inference
        if q_embeddings is None or p_embeddings is None:
            return q_embeddings, p_embeddings

        similarity = (
            torch.matmul(q_embeddings, p_embeddings.T) / self.temperature
        )  # (n_queries, n_passages)
        target = torch.arange(len(q_embeddings)).to(q_embeddings.device)  # (n_queries,)
        loss = F.cross_entropy(similarity, target)
        return loss, similarity

    def encode_text(
        self,
        texts: Union[str, List[str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length=128,
    ):
        if isinstance(texts, str):
            texts = [texts]
        texts = [tokenizer.bos_token + text + tokenizer.eos_token for text in texts]
        inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding_side="right",
            padding="max_length",
            return_token_type_ids=False,
        )
        embeddings = self.encode(inputs.to(self.encoder.device))
        return embeddings

    def encode(self, inputs: BatchEncoding):
        hidden_states = self.encoder(
            **inputs
        ).last_hidden_state  # (batch_size, seq_len, hidden_dim)
        return self.pooling(hidden_states, inputs["attention_mask"])

    def pooling(self, last_hidden_state, attention_mask):
        """
        last_hidden_state: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)
        embeddings: (batch_size, hidden_dim)
        """
        if isinstance(self.encoder, BertModel):
            last_hidden_state = last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        else:
            attention_mask = torch.where(
                attention_mask == 1,
                attention_mask
                + torch.arange(attention_mask.shape[1]).to(attention_mask.device),
                0,
            )  # (batch_size, seq_len)
            last_nonzero_indices = attention_mask.argmax(
                dim=1, keepdim=True
            )  # (batch_size, 1)
            last_token_indices = last_nonzero_indices.unsqueeze(-1).expand(
                -1, -1, last_hidden_state.shape[-1]
            )  # (batch_size, 1, hidden_dim)
            last_hidden_state = torch.gather(
                last_hidden_state, 1, last_token_indices
            )  # (batch_size, 1, hidden_dim)
            last_hidden_state = last_hidden_state.squeeze(1)  # (batch_size, hidden_dim)
        embeddings = F.normalize(last_hidden_state, p=2, dim=1)
        return embeddings


@torch.inference_mode()
def similarity_bert(
    model_bert: SentenceTransformer, queries: List[str], passages: List[str]
):
    if isinstance(queries, str):
        queries = [queries]
    if isinstance(passages, str):
        passages = [passages]
    q_embeddings = model_bert.encode(queries)  # (n_queries, hidden_dim)
    p_embeddings = model_bert.encode(passages)  # (n_passages, hidden_dim)
    similarity = model_bert.similarity(
        q_embeddings, p_embeddings
    )  # (n_queries, n_passages)
    target = torch.arange(len(q_embeddings)).to(q_embeddings.device)  # (n_queries,)
    loss = F.cross_entropy(similarity, target)
    return loss, similarity


@torch.inference_mode()
def calc_relative_advantage(similarity):
    """
    similarity: (n_queries, n_passages)
    """
    sorted_similarity = torch.sort(similarity, 1, descending=True)[0]
    second_best = sorted_similarity[:, 1]
    return (torch.diag(similarity) - second_best) / second_best


@torch.inference_mode()
def benchmark_retriever(
    model: Union[Literal["bm25"], Encoder],
    dataset: DPRDataset,
    top_k=[5, 20, 40, 60, 80, 100],
    batch_size=256,
):
    if not np.iterable(top_k):
        top_k = [top_k]
    max_top_k = max(top_k)
    top_k = sorted(top_k)
    queries, passages = dataset.tolist()
    unique_queries = []
    unique_passages = []
    for i in range(len(passages)):
        if passages[i] not in unique_passages:
            unique_queries.append(queries[i])
            unique_passages.append(passages[i])
    queries, passages = unique_queries, unique_passages
    if isinstance(model, str):
        if model == "bm25":
            import bm25s

            corpus_tokens = bm25s.tokenize(passages, stopwords="en")
            retriever = bm25s.BM25(k1=0.9, b=0.4)
            retriever.index(corpus_tokens)
            query_tokens = bm25s.tokenize(queries)
            indices = retriever.retrieve(query_tokens, k=max_top_k)[0]
        else:
            raise NotImplementedError
    elif isinstance(model, Encoder):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_q_embeddings = []
        all_p_embeddings = []
        device = model.encoder.device
        for query_inputs, passage_inputs in tqdm(loader):
            q_embeddings = model.encode(
                query_inputs.to(device)
            )  # (batch_size, hidden_dim)
            p_embeddings = model.encode(
                passage_inputs.to(device)
            )  # (batch_size, hidden_dim)
            all_q_embeddings.append(q_embeddings)
            all_p_embeddings.append(p_embeddings)
        q_embeddings = torch.cat(all_q_embeddings, dim=0)  # (n_queries, hidden_dim)
        p_embeddings = torch.cat(all_p_embeddings, dim=0)  # (n_passages, hidden_dim)
        similarity = q_embeddings @ p_embeddings.T  # (n_queries, n_passages)
        values, indices = torch.topk(similarity, k=max_top_k, dim=1)
        indices = indices.cpu().numpy()
    else:  # sentence-transformers
        similarity = similarity_bert(model, queries, passages)[1]
        values, indices = torch.topk(similarity, k=max_top_k, dim=1)
        indices = indices.cpu().numpy()
    targets = np.arange(len(indices))[:, None]
    accuracy = {}
    for _top_k in top_k:
        acc = np.any(indices[:, :_top_k] == targets, 1).mean().item()
        accuracy[_top_k] = acc
    # torch.cuda.empty_cache()
    return accuracy


class BM25:
    """
    bm25 = BM25(k1=0.9, b=0.4)
    bm.index(corpus)
    bm.retrieve(queries, top_k=100)
    """

    def __init__(self, pattern=r"(?u)\b\w\w+\b", k1=0.9, b=0.4):
        """
        The default `k1` and `b` are consistent with Karpukhin et al. (2020).
        """
        self.pattern = re.compile(pattern)
        self.k1 = k1
        self.b = b

    def split(self, text: str):
        text = text.lower()
        text = self.pattern.findall(text)
        return text

    def index(self, corpus: List[str]):
        self.chunk_lens = []
        self.freqs = []
        token_to_num_chunks = Counter()
        if isinstance(corpus, str):
            corpus = [corpus]
        for text in tqdm(corpus):
            tokens = self.split(text)
            chunk_len = len(tokens)
            freqs = {}
            for token in tokens:
                if token not in freqs:
                    freqs[token] = 1
                    token_to_num_chunks[token] += 1
                else:
                    freqs[token] += 1
            self.chunk_lens.append(chunk_len)
            self.freqs.append(freqs)
        self.num_chunks = len(self.chunk_lens)
        self.mean_chunk_len = np.mean(self.chunk_lens).item()
        self.token_to_num_chunks = dict(token_to_num_chunks)

    def retrieve(self, queries: List[str], top_k=100):
        if isinstance(queries, str):
            queries = [queries]
        results = []
        for query in tqdm(queries):
            result = self._retrieve(query, top_k)
            results.append(result)
        indices = np.stack([result[0] for result in results])
        scores = np.stack([result[1] for result in results])
        return indices, scores

    def _retrieve(self, query: str, top_k=100):
        if top_k is None:
            top_k = self.num_chunks
        assert top_k <= self.num_chunks
        query_tokens = self.split(query)
        scores = np.zeros(self.num_chunks, dtype=float)
        for chunk_index in range(self.num_chunks):
            score = self.score(query_tokens, chunk_index)
            scores[chunk_index] = score
        indices = np.argsort(-scores)[:top_k]
        scores = scores[indices]
        return indices, scores

    def score(self, query_tokens: List[str], chunk_index: int):
        chunk_len = self.chunk_lens[chunk_index]
        n = np.array(
            list(
                map(lambda token: self.token_to_num_chunks.get(token, 0), query_tokens)
            )
        )
        freqs = np.array(
            list(map(lambda token: self.freqs[chunk_index].get(token, 0), query_tokens))
        )
        IDF = self.calc_IDF(n)
        weight = freqs / (
            freqs + self.k1 * (1 - self.b + self.b * chunk_len / self.mean_chunk_len)
        )
        return np.sum(IDF * weight)

    def calc_IDF(self, n: int):
        """
        n: number of chunks containing the term
        """
        return np.log((self.num_chunks - n + 0.5) / (n + 0.5) + 1)


# ----- train retriever -----


@dataclass
class TrainArgs:
    save_dir: Union[str, Path] = project_dir / "checkpoints"  # 模型保存目录
    save_weight: str = "dpr"  # 保存权重的前缀名
    epochs: int = 1  # 训练轮数
    batch_size: int = 32
    learning_rate: float = 4e-6
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    device: str = None
    dtype: str = "bfloat16"
    num_workers: int = 0  # 数据加载线程数
    accumulation_steps: int = 1  # 梯度累积步数
    grad_clip: float = 1.0  # 梯度裁剪阈值
    log_interval: int = 100  # 日志打印间隔
    save_interval: int = 100  # 模型保存间隔
    hidden_size: int = 512  # 隐藏层维度
    num_hidden_layers: int = 8  # 隐藏层数量
    max_query_len: int = 64  # 问题最大长度
    max_passage_len: int = 128  # 单个document的最大长度，对于`natural-questions`数据集，建议设置为512
    use_amp: bool = None  # 使用自动混合精度
    temperature: float = 1.0
    from_weight: str = "pretrain"  # 基于哪个权重训练，为none则从头开始
    from_resume: int = 0  # 是否自动检测&续训（0=否，1=是）
    dataset: str = "ChineseSquad"  # 训练数据集（同时也是检索数据集），`ChineseSquad`或`natural-questions`
    use_wandb: bool = False
    use_swanlab: bool = False  # 使用swanlab时需要同时设置use_wandb=True, wandb会被替换为swanlab
    train_bert: bool = False  # 是否训练BERT-Mini，若是则加载BERT，则MiniMind模型和权重均失效
    wandb_entity: str = None
    wandb_project: str = "MiniMind"

    def __post_init__(self):
        assert self.dtype in ["bfloat16", "float16"]
        assert self.dataset in ["ChineseSquad", "natural-questions"]
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.use_amp is None:
            self.use_amp = "cuda" in self.device
        self.save_dir = Path(self.save_dir).as_posix()
        if self.train_bert:
            self.hidden_size = 256  # BERT-mini
        self.model_dir = project_dir / "out"
        if not self.model_dir.exists():
            self.model_dir = project_dir / "checkpoints"
        self.model_dir = self.model_dir.as_posix()
        self.wandb = wandb
        if self.use_swanlab:
            import swanlab

            self.wandb = swanlab


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(
    lm_config=None,
    from_weight="pretrain",
    tokenizer_path="../model",
    save_dir="../out",
    device="cuda",
    bert=False,
    eval_args=None,
):
    if eval_args is not None:
        tokenizer = AutoTokenizer.from_pretrained(eval_args.load_from)
        if "model" in eval_args.load_from:
            model = MiniMindForCausalLM(
                MiniMindConfig(
                    hidden_size=eval_args.hidden_size,
                    num_hidden_layers=eval_args.num_hidden_layers,
                    use_moe=bool(eval_args.use_moe),
                    inference_rope_scaling=eval_args.inference_rope_scaling,
                )
            )
            moe_suffix = "_moe" if eval_args.use_moe else ""
            ckp = f"{eval_args.save_dir}/{eval_args.weight}_{eval_args.hidden_size}{moe_suffix}.pth"
            model.load_state_dict(
                torch.load(ckp, map_location=eval_args.device), strict=True
            )
            if eval_args.lora_weight != "None":
                apply_lora(model)
                load_lora(
                    model,
                    f"{eval_args.save_dir}/lora/{eval_args.lora_weight}_{eval_args.hidden_size}.pth",
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                eval_args.load_from, trust_remote_code=True
            )
    else:
        if bert:
            model = AutoModel.from_pretrained("prajjwal1/bert-mini")
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = MiniMindForCausalLM(lm_config)
        model = Encoder(model, temperature=1.0)

        if from_weight != "none":
            moe_suffix = "_moe" if lm_config.use_moe else ""
            weight_path = (
                f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            )
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)

    Logger(
        f"所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    return model.to(device), tokenizer


def train_epoch(
    model: MiniMindForCausalLM,
    args: TrainArgs,
    lm_config: MiniMindConfig,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler,
    epoch: int,
    loader: DataLoader,
    iters: int,
    bar,
    start_step=0,
    use_wandb=False,
    spend_time=0,
):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    grad_norm = float("nan")
    adv = float("nan")
    wandb = args.wandb
    start_time = time.time()

    def get_spend_time():
        return spend_time + time.time() - start_time

    for step, (query_inputs, passage_inputs) in enumerate(loader, start=start_step + 1):
        query_inputs = query_inputs.to(args.device)
        passage_inputs = passage_inputs.to(args.device)

        with torch.autocast(device_type=args.device, enabled=args.use_amp, dtype=dtype):
            loss, similarity = model(query_inputs, passage_inputs)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        loss = loss.detach().cpu().numpy().item() * args.accumulation_steps
        lr = optimizer.param_groups[-1]["lr"]

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            grad_norm = grad_norm.detach().cpu().numpy().item()

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            torch.cuda.empty_cache()

            if use_wandb:
                adv = (
                    calc_relative_advantage(similarity)
                    .mean()
                    .detach()
                    .cpu()
                    .numpy()
                    .item()
                )
                wandb.log(
                    {
                        "train_step": (epoch * len(loader) + step)
                        // args.accumulation_steps,
                        "train/epoch": epoch + 1,
                        "train/loss": loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/adv": adv,
                        "train/time": get_spend_time(),
                    }
                )
        scheduler.step()
        if bar is not None:
            bar.update()
            bar.set_postfix_str(
                f"[{epoch+1}/{args.epochs}]loss={loss:.4f},adv={adv:.4f},grad_norm={grad_norm:.4f}"
            )

        if step % args.log_interval == 0 or step == iters - 1:
            Logger(f"Epoch:[{epoch+1}/{args.epochs},{step}/{iters}] loss:{loss:.4f}")

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            wandb_id = None
            if args.use_wandb:
                wandb_id = getattr(wandb.run, "id", None)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                wandb_id=wandb_id,
                save_dir=args.save_dir,
                spend_time=get_spend_time(),
            )
            model.train()
    return get_spend_time()


def train(
    args: TrainArgs,
    model: MiniMindForCausalLM = None,
    tokenizer: PreTrainedTokenizerBase = None,
    train_ds: DPRDataset = None,
    early_return=False,
):
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir)
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    # set in TrainArgs

    # ========== 4. 定义模型、数据、优化器 ==========
    if model is None:
        model, tokenizer = init_model(
            lm_config,
            args.from_weight,
            tokenizer_path=project_dir / "model",
            save_dir=args.model_dir,
            device=args.device,
            bert=args.train_bert,
        )
        # model = Encoder(model, temperature=args.temperature)
    num_params = get_num_params(model)
    if train_ds is None:
        dataset = load_dataset(args.dataset, args.max_query_len, args.max_passage_len)
        train_ds = DPRDataset(
            dataset["train"],
            tokenizer,
            max_query_len=args.max_query_len,
            max_passage_len=args.max_passage_len,
        )
        key = "validation" if args.dataset == "ChineseSquad" else "test"
        test_ds = DPRDataset(
            dataset[key],
            tokenizer,
            max_query_len=args.max_query_len,
            max_passage_len=args.max_passage_len,
        )

    if early_return:  # for debugging
        return model, tokenizer, train_ds, test_ds

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.GradScaler(args.device, enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=args.betas,
    )

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    spend_time = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        spend_time = ckp_data.get("spend_time", 0)

    if start_step > 0:  # 第一个epoch且存在检查点
        batch_sampler = SkipBatchSampler(
            train_sampler or range(len(train_ds)), args.batch_size, start_step + 1
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:  # 默认从头开始
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    iters = len(loader)
    max_step = args.epochs * iters
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_step)
    if start_step > 0:
        scheduler.load_state_dict(ckp_data["scheduler"])

    # ========== 6. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7. 配wandb ==========
    run = nullcontext()
    use_wandb = args.use_wandb and is_main_process()
    if use_wandb:
        wandb = args.wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-{round(num_params/1e6)}M-{args.save_weight}-epoch-{args.epochs}-batchsize-{args.batch_size}-lr-{args.learning_rate}"
        if wandb_id:
            config = None
        else:
            config = asdict(args)
            for key in [
                "device",
                "from_resume",
                "use_wandb",
                "wandb_project",
                "save_dir",
            ]:
                config.pop(key)
            config["max_step"] = max_step
            config["num_params"] = num_params
        wandb_kwargs = dict(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
            config=config,
        )
        if args.use_swanlab:
            wandb_kwargs.pop("entity")
        run = wandb.init(**wandb_kwargs)
        if args.use_swanlab:
            run = nullcontext()
        else:
            wandb.define_metric("train_step")
            wandb.define_metric("eval_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("eval/*", step_metric="eval_step")

    # ========== 8. 开始训练 ==========
    bar = tqdm(total=max_step, disable=not is_main_process())
    with run:
        for epoch in range(start_epoch, args.epochs):
            train_sampler and train_sampler.set_epoch(epoch)
            if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
                Logger(
                    f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
                )
                # train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
                _iters = iters + start_step + 1
                bar.update(start_step)
            else:  # 默认从头开始
                _iters = iters
            spend_time = train_epoch(
                model=model,
                args=args,
                lm_config=lm_config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                loader=loader,
                iters=_iters,
                start_step=start_step,
                bar=bar,
                use_wandb=use_wandb,
                spend_time=spend_time,
            )
    bar.close()


# ----- RAG -----


@dataclass
class EvalArgs:
    load_from: str = project_dir / "model"
    save_dir: str = "out"
    weight: str = "full_sft"
    lora_weight: str = "None"
    hidden_size: int = 768  # chat模型隐藏层维度(与retriever无关)
    num_hidden_layers: int = 16  # chat模型隐藏层数量(与retriever无关)
    use_moe: int = 0
    inference_rope_scaling: bool = False
    max_new_tokens: int = 8192
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 20  # 检索时返回的文档数量，reranker会从中选择最终答案
    rag: bool = True  # 是否开启RAG
    max_query_len: int = 64
    max_passage_len: int = 128
    historys: int = 0
    use_sbert_retriever: bool = True  # 使用sentence-transformer作为retriever，检索成功率会大幅提升
    device: str = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.save_dir = Path(self.save_dir)
        if not self.save_dir.is_absolute():
            self.save_dir = project_dir / self.save_dir
        if not self.save_dir.exists():
            self.save_dir = project_dir / "checkpoints"
        self.save_dir = self.save_dir.as_posix()
        if isinstance(self.load_from, Path):
            self.load_from = self.load_from.as_posix()


@torch.inference_mode()
def init_rag(
    device,
    documents: Union[List[str], datasets.Dataset] = None,
    retriever: Encoder = None,
    tokenizer: PreTrainedTokenizerBase = None,
    eval_args: EvalArgs = None,
):
    print("loading models and encoding documents, this may take a while...")
    if retriever is None or eval_args.use_sbert_retriever:
        retriever = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        encode_fn = lambda retriever: retriever.encode(
            documents, convert_to_tensor=True, show_progress_bar=True
        )
    else:

        def encode_fn(retriever: Encoder):
            all_p_embeddings = []
            batch_size = 128
            step = int(np.ceil(len(documents) / batch_size))
            for i in trange(step):
                docs = documents[i * batch_size : (i + 1) * batch_size]
                p_embeddings = retriever.encode_text(
                    docs, tokenizer, max_length=eval_args.max_passage_len
                )
                all_p_embeddings.append(p_embeddings)
            p_embeddings = torch.cat(
                all_p_embeddings, dim=0
            )  # (n_passages, hidden_dim)
            return p_embeddings

    retriever.to(device)
    retriever.eval()
    reranker = AutoModelForSequenceClassification.from_pretrained(
        "jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True
    )
    reranker.to(device)
    reranker.eval()
    # return retriever, reranker
    if documents is None:
        documents = load_dataset()
        documents = documents["train"]["context"]
    doc_embeddings = encode_fn(retriever)
    return retriever, reranker, doc_embeddings


@torch.inference_mode()
def retrieve(
    query: str,
    retriever: Union[SentenceTransformer, Encoder],
    reranker: PreTrainedModel,
    documents: List[str],
    doc_embeddings: torch.Tensor,
    top_k=20,
    tokenizer=None,
    max_length=128,
):
    """
    corpus_embeddings: (n_passages, hidden_dim)
    """
    if isinstance(retriever, SentenceTransformer):
        query_embeddings = retriever.encode(
            query, convert_to_tensor=True
        )  # (hidden_dim,)
    else:
        query_embeddings = retriever.encode_text(
            query, tokenizer, max_length=max_length
        )[0]
    similarity = doc_embeddings @ query_embeddings
    values, indices = torch.topk(similarity, k=top_k)
    indices = indices.cpu().numpy()
    sentence_pairs = [[query, documents[i]] for i in indices]
    scores = reranker.compute_score(sentence_pairs)
    max_index = indices[np.argsort(scores)[-1]]
    return max_index.item(), documents[max_index]


@torch.inference_mode()
def evaluate(
    args: EvalArgs,
    retriever: SentenceTransformer,
    reranker: PreTrainedModel,
    documents: List[str],
    doc_embeddings: torch.Tensor,
):
    prompts = [
        "估计有4.88亿至5.35亿人信奉什么宗教",
        "南安普敦机场在哪个城镇",
        "路德什么时候死的",
        "超级碗开幕之夜于何时在何地举行",
    ]

    conversation = []
    model, tokenizer = init_model(eval_args=args)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)  # or setup_seed(random.randint(0, 2048))
        if input_mode == 0:
            print(f"👶: {prompt}")
        if args.rag:
            doc = retrieve(
                prompt,
                retriever,
                reranker,
                documents,
                doc_embeddings,
                top_k=args.top_k,
                tokenizer=tokenizer,
                max_length=args.max_passage_len,
            )[1]
            prompt = f'你是一个智能问答助手，请严格按照以下要求回答问题： 如果资料中包含问题答案，请直接使用资料信息并注明"根据资料"，如果资料不相关、信息不足或未包含答案，请明确说明"资料中未包含相关信息"，然后可以基于常识进行补充\n资料：{doc}\n问题：{prompt}\n现在请开始回答：'
        conversation = conversation[-args.historys :] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if args.weight == "reason":
            templates["enable_thinking"] = True  # 仅Reason模型使用
        inputs = (
            tokenizer.apply_chat_template(**templates)
            if args.weight != "pretrain"
            else (tokenizer.bos_token + prompt)
        )
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print("🤖️:", end="")
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})
        print("")
