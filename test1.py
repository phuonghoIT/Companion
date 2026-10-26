import torch  # pyright: ignore[reportMissingImports]

with open("training_data.txt", "r", encoding="utf-8") as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Danh sách ký tự: {''.join(chars)}")

#string to interger
stoi = {ch: i for i, ch in enumerate(chars)}
#integer to string
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype =torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block = 8





  


