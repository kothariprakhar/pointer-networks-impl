import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Assuming the code is saved in solution.py
# from solution import PointerNetwork, collate_fn, SortingDataset

# For the purpose of this self-contained test suite, we re-import or expect the classes to be available.
# In a real environment, you would import them. 
# We will replicate the minimal necessary imports to make this test runnable if classes are in the namespace.

class TestPointerNetwork(unittest.TestCase):

    def setUp(self):
        # Model Hyperparameters
        self.input_dim = 1
        self.hidden_dim = 32
        self.device = torch.device('cpu')
        self.model = PointerNetwork(self.input_dim, self.hidden_dim).to(self.device)

    def test_output_shape(self):
        """Ensure output shape is (Batch, SeqLen, SeqLen_Logits)."""
        batch_size = 4
        seq_len = 10
        # Create random input (Batch, SeqLen, InputDim)
        x = torch.randn(batch_size, seq_len, self.input_dim)
        # Create dummy mask (all valid)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        output = self.model(x, mask=mask)
        
        # Expected: (Batch, SeqLen, SeqLen)
        # Note: In Pointer Nets, the output vocabulary size is the sequence length itself.
        self.assertEqual(output.shape, (batch_size, seq_len, seq_len))

    def test_masking_logic(self):
        """Ensure masked positions have -inf scores."""
        # Batch of 2 items: One length 5, One length 3 (padded to 5)
        # We simulate the padding scenario manually
        x = torch.randn(2, 5, self.input_dim)
        
        # Mask: First sample all True, Second sample T, T, T, F, F
        mask = torch.tensor([
            [True, True, True, True, True],
            [True, True, True, False, False]
        ], dtype=torch.bool)
        
        output = self.model(x, mask=mask)
        
        # Check second sample (index 1)
        # The logits corresponding to indices 3 and 4 should be -inf (or very small)
        # Output shape: (Batch, OutputStep, LogitIndex)
        # We check the logits for the first decoding step
        sample_2_logits = output[1, 0, :]
        
        self.assertTrue(sample_2_logits[3] < -1e5, "Masked index 3 should be -inf")
        self.assertTrue(sample_2_logits[4] < -1e5, "Masked index 4 should be -inf")
        self.assertTrue(sample_2_logits[0] > -1e5, "Valid index 0 should be normal")

    def test_teacher_forcing_safety(self):
        """Ensure teacher forcing handles padding targets (-1) without crashing."""
        x = torch.randn(2, 5, self.input_dim)
        mask = torch.ones(2, 5, dtype=torch.bool)
        
        # Targets with -1 padding
        targets = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, -1, -1] # This sample effectively ends at step 3
        ], dtype=torch.long)
        
        try:
            # Force teacher forcing
            _ = self.model(x, targets=targets, mask=mask, teacher_forcing_ratio=1.0)
        except IndexError:
            self.fail("Teacher forcing raised IndexError on padding target -1")

    def test_gradient_flow(self):
        """Ensure gradients are computed and backward pass runs."""
        x = torch.randn(2, 5, self.input_dim)
        mask = torch.ones(2, 5, dtype=torch.bool)
        targets = torch.randint(0, 5, (2, 5))
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        outputs = self.model(x, targets=targets, mask=mask)
        
        # Flatten for loss
        loss = nn.CrossEntropyLoss()(outputs.reshape(-1, 5), targets.reshape(-1))
        loss.backward()
        
        # Check if embedding weights have gradient
        self.assertIsNotNone(self.model.embedding.weight.grad)
        optimizer.step()

    def test_collate_fn(self):
        """Test if collate function produces correct shapes and masks."""
        batch = [
            {'input': torch.randn(5, 1), 'target': torch.tensor([0, 1, 2, 3, 4])},
            {'input': torch.randn(3, 1), 'target': torch.tensor([0, 1, 2])}
        ]
        
        padded_inputs, padded_targets, mask = collate_fn(batch)
        
        # Max len should be 5
        self.assertEqual(padded_inputs.shape, (2, 5, 1))
        self.assertEqual(padded_targets.shape, (2, 5))
        self.assertEqual(mask.shape, (2, 5))
        
        # Check mask content
        # First sample: all True
        self.assertTrue(torch.all(mask[0]))
        # Second sample: T, T, T, F, F
        self.assertTrue(torch.all(mask[1, :3]))
        self.assertFalse(torch.any(mask[1, 3:]))
        
        # Check padding values
        # Target padding should be -1
        self.assertEqual(padded_targets[1, 3], -1)
        self.assertEqual(padded_targets[1, 4], -1)

if __name__ == '__main__':
    unittest.main()