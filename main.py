import os
import psycopg2
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from torch.nn import functional as F

app = FastAPI()

# --- إعدادات قاعدة البيانات ---
DATABASE_URL = os.getenv("DATABASE_URL")

def save_to_db(prompt: str, language: str, code: str):
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO generated_codes (prompt, language, code) VALUES (%s, %s, %s);",
            (prompt, language, code)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database Error: {e}")

# --- محرك Transformer المصغر ---
d_model = 32
block_size = 16
n_head = 2
n_layer = 2
device = 'cpu'

# نصوص التدريب الأولية للنموذج
train_corpus = (
    "def calc(a, b): return a + b\n"
    "def sub(a, b): return a - b\n"
    "print('Hello Cloud AI')\n"
)
chars = sorted(list(set(train_corpus)))
if ' ' not in chars:
    chars.append(' ')
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        head_size = d_model // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# تهيئة وتدريب النموذج عند بدء تشغيل السيرفر
model = MiniLLM(vocab_size).to(device)
raw_data = torch.tensor(encode(train_corpus), dtype=torch.long)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for _ in range(120):
    ix = torch.randint(len(raw_data) - block_size, (2,))
    x = torch.stack([raw_data[i:i+block_size] for i in ix])
    y = torch.stack([raw_data[i+1:i+block_size+1] for i in ix])
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- نقاط الاتصال API ---
class RequestBody(BaseModel):
    prompt: str
    language: str

@app.get("/")
def read_root():
    return {"status": "AI Transformer Engine Online"}

@app.post("/generate")
def generate_code(req: RequestBody):
    start_tokens = encode(req.prompt[:block_size])
    if not start_tokens:
        start_tokens = [0]

    input_tensor = torch.tensor([start_tokens], dtype=torch.long, device=device)
    output_tokens = model.generate(input_tensor, max_new_tokens=50)[0].tolist()
    generated_text = decode(output_tokens)

    # توثيق وحفظ المخرجات في PostgreSQL
    save_to_db(req.prompt, req.language, generated_text)

    return {"generated_code": generated_text}
