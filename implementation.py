import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Data Strategy: The Sorting Task
# ==========================================

class SortingDataset(Dataset):
    """
    Generates sequences of random integers.
    Input: [0.5, 0.1, 0.3]
    Target (Indices pointing to sorted values): [1, 2, 0] -> (0.1, 0.3, 0.5)
    """
    def __init__(self, sample_size, min_len, max_len):
        self.sample_size = sample_size
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # Random length for this sample
        length = np.random.randint(self.min_len, self.max_len + 1)
        
        # Generate random input sequence (0-1 floats)
        input_seq = np.random.rand(length)
        
        # Get the argsort indices (the labels)
        target_indices = np.argsort(input_seq)
        
        return {
            'input': torch.FloatTensor(input_seq).unsqueeze(1), # (Seq_Len, 1)
            'target': torch.LongTensor(target_indices)
        }

def collate_fn(batch):
    """
    Custom collate to handle variable length sequences via padding.
    """
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Pad sequences
    # Inputs padded with 0.0
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)
    # Targets padded with -1 (ignore_index)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
    
    # Create a mask for valid inputs (Batch, SeqLen)
    # We calculate mask based on original lengths to be robust against 0.0 values in data
    lengths = torch.tensor([len(x) for x in inputs])
    max_len = inputs_padded.size(1)
    # Mask: True if valid, False if padding
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    return inputs_padded, targets_padded, mask

# ==========================================
# 2. Model Architecture: Pointer Network
# ==========================================

class Attention(nn.Module):
    """
    Bahdanau-style Additive Attention.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_state, mask):
        # encoder_outputs: (Batch, SeqLen, Hidden)
        # decoder_state: (Batch, Hidden)
        # mask: (Batch, SeqLen) - True for valid tokens, False for padding
        
        # Project features
        encoder_proj = self.W1(encoder_outputs)       # (Batch, SeqLen, Hidden)
        decoder_proj = self.W2(decoder_state).unsqueeze(1) # (Batch, 1, Hidden)
        
        # Additive Attention Score
        u = torch.tanh(encoder_proj + decoder_proj)
        scores = self.V(u).squeeze(-1) # (Batch, SeqLen)
        
        # Apply Mask
        # We set padding locations to -inf so softmax/pointer prob is 0
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores

class PointerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PointerNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 2. Encoder
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Decoder
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # 4. Pointer Attention
        self.attention = Attention(hidden_dim)
        
        # Learnable Initial Input for Decoder
        self.start_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, targets=None, mask=None, teacher_forcing_ratio=0.5):
        """
        x: (Batch, SeqLen, InputDim)
        targets: (Batch, SeqLen) indices.
        mask: (Batch, SeqLen) Boolean mask.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # === Encoding ===
        embedded_inputs = self.embedding(x)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_inputs)
        
        # Decoder initial state
        decoder_hidden = hidden.squeeze(0)
        decoder_cell = cell.squeeze(0)
        
        # Initial decoder input
        decoder_input = self.start_token.unsqueeze(0).repeat(batch_size, 1)
        
        outputs = []
        
        # === Decoding Loop ===
        for t in range(seq_len):
            # 1. Run Decoder Cell
            decoder_hidden, decoder_cell = self.decoder_cell(decoder_input, (decoder_hidden, decoder_cell))
            
            # 2. Compute Pointer Logits
            scores = self.attention(encoder_outputs, decoder_hidden, mask)
            outputs.append(scores)
            
            # 3. Select Next Input (Greedy or Teacher Forcing)
            _, preds = torch.max(scores, dim=1)
            
            if targets is not None and random.random() < teacher_forcing_ratio:
                # Teacher Forcing
                current_target = targets[:, t] # (Batch,)
                
                # Fix: Handle padding indices (-1) in targets
                # If target is -1, we clamp it to 0 (or any valid index) just for the gather op.
                # The loss at this position is masked out anyway.
                safe_indices = current_target.clone()
                safe_indices[safe_indices == -1] = 0
                
                next_input_idx = safe_indices.unsqueeze(1)
            else:
                # No Teacher Forcing
                next_input_idx = preds.unsqueeze(1)

            # Gather embedding for next step
            # embedded_inputs: (B, L, H)
            # next_input_idx: (B, 1)
            next_input_embedding = torch.gather(
                embedded_inputs, 
                1, 
                next_input_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )
            decoder_input = next_input_embedding.squeeze(1)

        outputs = torch.stack(outputs, dim=1) # (Batch, SeqLen, SeqLen_Logits)
        return outputs

# ==========================================
# 3. Training Routine
# ==========================================

def train():
    INPUT_DIM = 1
    HIDDEN_DIM = 256
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    
    train_dataset = SortingDataset(3000, 5, 10)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = PointerNetwork(INPUT_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Ignore index -1 corresponds to padding in targets
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print("Starting Training on Sorting Task...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for inputs, targets, mask in train_loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Pass mask to forward
            outputs = model(inputs, targets=targets, mask=mask, teacher_forcing_ratio=0.5)
            
            # Flatten for loss: (B*L, L_logits) vs (B*L)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

    return model

# ==========================================
# 4. Evaluation
# ==========================================

def evaluate(model):
    model.eval()
    print("\n--- Evaluation: Generalization Test ---")
    
    test_len = 15
    input_seq = np.random.rand(test_len)
    
    # Prepare single batch
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(1).unsqueeze(0).to(device)
    mask = torch.ones(1, test_len, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        # No teacher forcing during eval
        logits = model(input_tensor, mask=mask, teacher_forcing_ratio=0.0)
        _, predictions = torch.max(logits, dim=2)
        predictions = predictions.squeeze(0).cpu().numpy()

    print(f"Input (first 5): {np.round(input_seq[:5], 2)}...")
    predicted_values = input_seq[predictions]
    is_sorted = np.all(predicted_values[:-1] <= predicted_values[1:])
    
    print(f"Correct Indices:   {np.argsort(input_seq)}")
    print(f"Predicted Indices: {predictions}")
    print(f"Successfully Sorted? {is_sorted}")

if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)
