import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. CÁC SIÊU THAM SỐ (HYPERPARAMETERS)
# ==============================================================================
   # Tỉ lệ tắt ngẫu nhiên nơ-ron để chống học vẹt (overfitting)

# Cấu hình tối ưu cho CPU Mac 2015
batch_size = 32         # Số mẩu văn bản xử lý cùng lúc
block_size = 64         # Độ dài ngữ cảnh (Context Length)
max_iters = 3000        # Số bước huấn luyện
eval_interval = 250     # <--- DÒNG BỊ THIẾU CẦN THÊM VÀO ĐÂY
eval_iters = 40         # Số lần lấy mẫu để tính trung bình loss
learning_rate = 5e-4    # Tốc độ học (0.0005)
n_embd = 64             # Số chiều vector biểu diễn mỗi token
n_head = 4              # Số đầu chú ý
n_layer = 4             # Số tầng Transformer Block
dropout = 0.1           # Tỉ lệ dropout chống học vẹt

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1337)
torch.manual_seed(1337)

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU & TOKENIZER (DÙNG ĐÚNG TIN NHẮN THẬT)
# ==============================================================================
with open("training_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Lập bảng chữ cái (Vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Bộ chuyển đổi: Ký tự <-> Số nguyên
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# Chuyển toàn bộ dữ liệu thành Tensor và chia tập Train/Val (90/10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Bốc ngẫu nhiên một mẻ (batch) bài tập (x) và đáp án (y)"""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """Hàm ước lượng loss trung bình trên cả tập train và val mà không tính đạo hàm"""
    out = {}
    model.eval() # Chuyển mô hình sang chế độ đánh giá (tắt dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Bật lại chế độ huấn luyện
    return out

# ==============================================================================
# 3. CÁC KHỐI KIẾN TRÚC TRANSFORMER
# ==============================================================================

class Head(nn.Module):
    """Một đầu Self-Attention đơn lẻ: Q, K, V và Causal Mask"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Mặt nạ tam giác dưới che tương lai
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # 1. Tính Attention Scores: Q @ K^T / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        # 2. Úp mặt nạ che tương lai
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 3. Chuẩn hóa xác suất bằng Softmax
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # 4. Gom thông tin từ Value
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Nhiều đầu Attention chạy song song để nhìn ngữ cảnh đa chiều"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd) # Chiếu tuyến tính sau khi gộp
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Ghép nối (concatenate) kết quả của tất cả các đầu theo chiều sâu
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Mạng Nơ-ron truyền xuôi đơn giản: Từng từ tự ngẫm nghĩ độc lập"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Mở rộng không gian đặc trưng gấp 4 lần
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Thu về kích thước ban đầu
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Một khối Transformer hoàn chỉnh: Giao tiếp (Attention) -> Suy ngẫm (FFN)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Đường truyền tắt (Residual connection) x + ... kèm chuẩn hóa tiền tầng (Pre-LayerNorm)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ==============================================================================
# 4. MÔ HÌNH GPT HOÀN CHỈNH
# ==============================================================================

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Bảng tra cứu ý nghĩa ký tự & Bảng tra cứu vị trí
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Xếp chồng nhiều khối Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Chuẩn hóa tầng cuối cùng
        
        # Tầng tuyến tính chiếu từ n_embd ra xác suất của từng ký tự trong từ điển
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Khởi tạo trọng số chuẩn theo phong cách OpenAI GPT
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

        # Token Embedding + Position Embedding
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Cho dữ liệu chạy qua tất cả các khối Transformer Blocks
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Hàm tự sinh ký tự mới nối tiếp ngữ cảnh"""
        for _ in range(max_new_tokens):
            # Cắt ngữ cảnh để không vượt quá block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Chỉ lấy điểm số của ký tự cuối cùng
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Bốc thăm có trọng số xác suất
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# 5. KHỞI TẠO VÀ HUẤN LUYỆN
# ==============================================================================
model = GPTLanguageModel().to(device)
print(f"Mô hình đang chạy trên thiết bị: {device}")
print(f"Tổng số tham số (Parameters): {sum(p.numel() for p in model.parameters())/1e3:.2f}K params")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- BẮT ĐẦU HUẤN LUYỆN MINI-GPT ---")
for iter in range(max_iters):
    # Định kỳ đánh giá độ chính xác trên cả 2 tập dữ liệu
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Bước {iter:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    # Lấy dữ liệu và thực hiện một bước học
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==============================================================================
# 6. SINH VĂN BẢN THỬ NGHIỆM
# ==============================================================================
print("\n" + "="*50)
print("KẾT QUẢ MÔ HÌNH TỰ SINH VĂN BẢN (CHUYỂN ĐỘNG THEO TIN NHẮN THẬT):")
print("="*50)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=400)[0].tolist())
print(generated_text)