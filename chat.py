import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. CÁC SIÊU THAM SỐ (TỐI ƯU HÓA CHO CPU INTEL TRÊN MAC)
# ==============================================================================
batch_size = 32         # Số đoạn text xử lý song song
block_size = 64         # Độ dài ngữ cảnh tối đa (64 ký tự)
max_iters = 2500        # Số bước huấn luyện
eval_interval = 250     # Số bước kiểm tra loss định kỳ
learning_rate = 5e-4    # Tốc độ học của AdamW
eval_iters = 40         # Số lượt batch dùng để ước lượng loss
n_embd = 64             # Kích thước vector nhúng token
n_head = 4              # Số lượng đầu chú ý Multi-Head
n_layer = 4             # Số lượng khối Transformer Blocks
dropout = 0.1           # Tỉ lệ dropout

# Thiết bị tính toán: CPU đối với Mac Intel 2015
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1337)

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU & TOKENIZER
# ==============================================================================
data_path = "training_data.txt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy file {data_path}. Vui lòng kiểm tra lại đường dẫn!")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Lập bảng chữ cái (Vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Từ điển ký tự <-> số nguyên
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# Chuyển toàn bộ dữ liệu thành Tensor và chia Train/Val (90/10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==============================================================================
# 3. CÁC MODULE KIẾN TRÚC TRANSFORMER
# ==============================================================================

class Head(nn.Module):
    """Một đầu Self-Attention đơn lẻ: Q, K, V và Causal Mask"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Tính toán ma trận Attention Scores: (Q @ K^T) / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Gộp nhiều đầu Attention chạy song song"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Mạng tuyến tính truyền xuôi độc lập cho từng vị trí"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Một khối Transformer: Giao tiếp (Attention) -> Suy ngẫm (FFN)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7):
        """Sinh chuỗi ký tự tiếp theo"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# 4. KHỞI TẠO VÀ TIẾN HÀNH HUẤN LUYỆN
# ==============================================================================
model = GPTLanguageModel().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Thiết bị đang sử dụng: {device}")
print(f"Tổng số tham số: {num_params/1e3:.2f}K params")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- BẮT ĐẦU HUẤN LUYỆN MINI-GPT ---")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Bước {iter:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==============================================================================
# 5. CHẾ ĐỘ TRÒ CHUYỆN TRỰC TIẾP (INTERACTIVE CHAT)
# ==============================================================================
print("\n" + "="*50)
print("HUẤN LUYỆN HOÀN TẤT! BẮT ĐẦU CHAT VỚI COMPANION")
print("(Gõ 'exit' hoặc 'quit' để kết thúc)")
print("="*50)

model.eval()

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Tạm biệt!")
        break
    
    if not user_input.strip():
        continue

    # Chuẩn bị prompt dạng hội thoại
    prompt = f"\nYou: {user_input}\nHer:"
    encoded_prompt = encode(prompt)
    
    if not encoded_prompt:
        print("Her: ... (Tin nhắn chứa ký tự không có trong từ điển)")
        continue

    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    # Sinh phản hồi
    with torch.no_grad():
        generated_indices = model.generate(context, max_new_tokens=80, temperature=0.7)[0].tolist()

    full_response = decode(generated_indices)
    her_reply = full_response[len(prompt):]

    # Cắt nếu mô hình tự nhảy sang lượt của You:
    if "\nYou:" in her_reply:
        her_reply = her_reply.split("\nYou:")[0]

    print(f"Her:{her_reply.strip()}")