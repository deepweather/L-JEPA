from typing import Any
import torch
import torch.nn as nn
from torch.autograd import profiler

class ModelAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, model):
        self.model = model
        self.model_params_memory = self.calculate_model_params_memory()
        self.flops_estimate = self.estimate_flops()
        self.training_memory = self.calculate_training_memory()
        self.max_seq_length = self.calculate_max_seq_length()
        self.max_batch_size = self.calculate_max_batch_size()
        self.report()
        return self

    def calculate_model_params_memory(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        memory = total_params * 4 s
        return memory / (1024 ** 2)

    def estimate_flops(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 2 * total_params

    def calculate_training_memory(self):
        return 3 * self.model_params_memory 

    def calculate_max_seq_length(self):
        seq_length = 8
        d_model = next(self.model.parameters()).size(-1)
        while True:
            try:
                dummy_src = torch.randn(1, seq_length, d_model, dtype=torch.float32).to(self.device)
                dummy_tgt = torch.randn(1, seq_length, d_model, dtype=torch.float32).to(self.device)
                self.model.to(self.device)
                self.model(dummy_src, dummy_tgt)
                seq_length *= 2
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return seq_length // 2

    def calculate_max_batch_size(self):
        batch_size = 1
        seq_length = 8
        d_model = next(self.model.parameters()).size(-1) 
        while True:
            try:
                dummy_src = torch.randn(batch_size, seq_length, d_model, dtype=torch.float32).to(self.device)
                dummy_tgt = torch.randn(batch_size, seq_length, d_model, dtype=torch.float32).to(self.device)
                self.model.to(self.device)
                self.model(dummy_src, dummy_tgt)
                batch_size *= 2
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return batch_size // 2 

    def report(self, to_file=None):
        report_str = f"""
        Model Analysis Report:
        ----------------------
        Trainable Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}
        Model Parameters Memory (MB): {self.model_params_memory:.2f}
        Estimated FLOPs (forward pass): {self.flops_estimate:,}
        Training Memory (MB): {self.training_memory:.2f}
        Max Sequence Length (before OOM): {self.max_seq_length}
        Max Batch Size (before OOM, at seq length 8): {self.max_batch_size}
        """
        print(report_str)
        
        if to_file:
            with open(to_file, 'w') as f:
                f.write(report_str)

    def clear_cache(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer(nn.Transformer(d_model=1024, nhead=16, num_encoder_layers=8, num_decoder_layers=8))
