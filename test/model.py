import argparse
import pickle
import string
from typing import Iterable, Tuple

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
import math 
import random
from torch.utils.data import Subset

ALPHABET = string.ascii_lowercase + string.digits + "."
char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx) + 1

MAX_LEN = 50


class DomainDataset(Dataset):
    def __init__(self, samples: Iterable[Tuple[str, int]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        dom, lbl = self.samples[idx]
        x = domain_to_tensor(dom)
        return x, lbl


def domain_to_tensor(domain: str) -> torch.Tensor:
    arr = [char2idx.get(c, 0) for c in domain.lower()][:MAX_LEN]
    arr += [0] * (MAX_LEN - len(arr))
    return torch.tensor(arr, dtype=torch.long)


def tensor_to_domain(tensor: torch.Tensor) -> str:
    domain = "".join(idx2char.get(idx, "") for idx in tensor.tolist() if idx > 0)
    return domain


def load_dataset(file_path: str) -> Dataset:
    with open(file_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def take_subset(dataset: Dataset, n: int) -> Subset:
    n = min(n, len(dataset))
    return Subset(dataset, list(range(n)))

def balance_dataset(ds, ratio=0.5):
    """Return a balanced subset with roughly 50/50 labels."""
   

    # Extract indices by label
    idx_0 = [i for i, (_, y) in enumerate(ds) if int(y) == 0]
    idx_1 = [i for i, (_, y) in enumerate(ds) if int(y) == 1]

    # Choose equal size for both classes
    n = min(len(idx_0), len(idx_1))
    random.shuffle(idx_0)
    random.shuffle(idx_1)
    selected = idx_0[:n] + idx_1[:n]
    random.shuffle(selected)
    return Subset(ds, selected)


def vqc_block(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights) : 
        for i in range (n_qubits) : 
            qml.Hadamard(wires=i)
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for wire in range(n_qubits):
            
            qml.RY(weights[0, wire], wires=wire)
            qml.RZ(weights[1, wire], wires=wire)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (2, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.vqc = vqc_block(n_qubits)
        self.output_linear = nn.Linear(n_qubits, hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h_prev], dim=1)

        q_inputs = self.input_linear(combined)
        q_inputs = math.pi * torch.tanh(q_inputs) # limit the angle 
        q_features = self.vqc(q_inputs)
        return self.output_linear(q_features)


class QuantumLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim        
        self.forget_gate = QuantumGate(input_dim, hidden_dim, n_qubits)
        self.input_gate = QuantumGate(input_dim, hidden_dim, n_qubits)
        self.candidate_gate = QuantumGate(input_dim, hidden_dim, n_qubits)
        self.output_gate = QuantumGate(input_dim, hidden_dim, n_qubits)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        f_t = torch.sigmoid(self.forget_gate(x_t, h_prev))
        i_t = torch.sigmoid(self.input_gate(x_t, h_prev))
        g_t = torch.tanh(self.candidate_gate(x_t, h_prev))
        c_t = f_t * c_prev + i_t * g_t
        o_t = torch.sigmoid(self.output_gate(x_t, h_prev))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class QuantumLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_qubits: int, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell = QuantumLSTMCell(embed_dim, hidden_dim, n_qubits)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        batch_size,T ,E = emb.shape 
        h_t = emb.new_zeros((batch_size, self.cell.hidden_dim))
        c_t = emb.new_zeros((batch_size, self.cell.hidden_dim))
        hs =[]
        for t in range(emb.size(1)):
            h_t, c_t = self.cell(emb[:, t, :], (h_t, c_t))
            hs.append(h_t.unsqueeze(1))
        H_seq = torch.cat(hs,dim=1) 
        mask = (x!=0).float().unsqueeze(-1)   
        pooled = (H_seq*mask ).sum(dim= 1 )/(mask.sum(dim =1 )+1e-9 )
        logits = self.fc ( self.dropout (pooled ))
        
        return logits 


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in tqdm(loader, desc="train"):
        if x_batch.size(0) == 1:
            continue
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)
    return running_loss / max(len(loader.dataset), 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return accuracy, precision, recall, f1


# def build_loaders(data_dir: str, batch_size: int, train_size: int, test_size: int):
#     benign_train_ds = load_dataset(f"{data_dir}/benign_train.pkl")
#     benign_test_ds = load_dataset(f"{data_dir}/benign_test.pkl")
#     dga_1_train_ds = load_dataset(f"{data_dir}/dga_1_train.pkl")
#     dga_1_test_ds = load_dataset(f"{data_dir}/dga_1_test.pkl")
#     dga_2_train_ds = load_dataset(f"{data_dir}/dga_2_train.pkl")
#     dga_2_test_ds = load_dataset(f"{data_dir}/dga_2_test.pkl")
#     dga_3_train_ds = load_dataset(f"{data_dir}/dga_3_train.pkl")
#     dga_3_test_ds = load_dataset(f"{data_dir}/dga_3_test.pkl")
#     dga_4_train_ds = load_dataset(f"{data_dir}/dga_4_train.pkl")
#     dga_4_test_ds = load_dataset(f"{data_dir}/dga_4_test.pkl")

#     train_ds = ConcatDataset([
#         benign_train_ds,
#         dga_1_train_ds,
#         dga_2_train_ds,
#         dga_3_train_ds,
#         dga_4_train_ds,
#     ])
#     test_ds = ConcatDataset([
#         benign_test_ds,
#         dga_1_test_ds,
#         dga_2_test_ds,
#         dga_3_test_ds,
#         dga_4_test_ds,
#     ])

#     if train_size > 0:
#         train_ds = take_subset(train_ds, train_size)
#     if test_size > 0:
#         test_ds = take_subset(test_ds, test_size)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader
def build_loaders(data_dir: str, batch_size: int, train_size: int, test_size: int):
   
    benign_train_ds = load_dataset(f"{data_dir}/benign_train.pkl")
    benign_test_ds  = load_dataset(f"{data_dir}/benign_test.pkl")
    dga_1_train_ds  = load_dataset(f"{data_dir}/dga_1_train.pkl")
    dga_1_test_ds   = load_dataset(f"{data_dir}/dga_1_test.pkl")
    dga_2_train_ds  = load_dataset(f"{data_dir}/dga_2_train.pkl")
    dga_2_test_ds   = load_dataset(f"{data_dir}/dga_2_test.pkl")
    dga_3_train_ds  = load_dataset(f"{data_dir}/dga_3_train.pkl")
    dga_3_test_ds   = load_dataset(f"{data_dir}/dga_3_test.pkl")
    dga_4_train_ds  = load_dataset(f"{data_dir}/dga_4_train.pkl")
    dga_4_test_ds   = load_dataset(f"{data_dir}/dga_4_test.pkl")

    train_ds = ConcatDataset([
        benign_train_ds,
        dga_1_train_ds,
        dga_2_train_ds,
        dga_3_train_ds,
        dga_4_train_ds,
    ])
    test_ds = ConcatDataset([
        benign_test_ds,
        dga_1_test_ds,
        dga_2_test_ds,
        dga_3_test_ds,
        dga_4_test_ds,
    ])

   
    train_ds = balance_dataset(train_ds)
    test_ds  = balance_dataset(test_ds)

    if train_size > 0:
        train_ds = take_subset(train_ds, train_size)
    if test_size > 0:
        test_ds = take_subset(test_ds, test_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def parse_args():
    parser = argparse.ArgumentParser(description="Quantum LSTM classifier with PennyLane")
    parser.add_argument("--data-dir", default="domain2", help="Path to directory containing pickled datasets")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=80000, help="Number of training samples to use (0 for full)")
    parser.add_argument("--test-size", type=int, default=0, help="Number of test samples to use (0 for full)")
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.train_size, args.test_size)
        # Đếm số mẫu từng nhãn trong tập train và test
    from collections import Counter

    all_train_labels = []
    for _, lbl in train_loader.dataset:
        if isinstance(lbl, torch.Tensor):
            all_train_labels.append(int(lbl.item()))
        else:
            all_train_labels.append(int(lbl))
    train_counts = Counter(all_train_labels)

    all_test_labels = []
    for _, lbl in test_loader.dataset:
        if isinstance(lbl, torch.Tensor):
            all_test_labels.append(int(lbl.item()))
        else:
            all_test_labels.append(int(lbl))
    test_counts = Counter(all_test_labels)

    print(f"Train set: {train_counts[0]} samples with label 0, {train_counts[1]} samples with label 1")
    print(f"Test set:  {test_counts[0]} samples with label 0, {test_counts[1]} samples with label 1")
    


    model = QuantumLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_qubits=args.n_qubits,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, prec, rec, f1 = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} | Accuracy={acc:.4f}, Precision={prec:.4f}, "
            f"Recall={rec:.4f}, F1-score={f1:.4f}"
        )


if __name__ =="__main__":
    main()
# if __name__ == "__main__":
    
#  
