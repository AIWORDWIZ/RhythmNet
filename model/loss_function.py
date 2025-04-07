from typing import Any, Tuple
import torch
import torch.nn as nn


class SmoothLoss(torch.autograd.Function):
    """
    Custom autograd function implementing a smoothness loss for heart rate predictions.
    This loss penalizes deviations from the mean heart rate in a sequence.
    """
    @staticmethod
    def forward(ctx, hr_t, hr_seq, T):
        """
        Forward pass for the smoothness loss.

        Args:
            ctx: context object that can be used to stash information
                 for backward computation
            hr_t: current heart rate value
            hr_seq: sequence of heart rate values
            T: number of elements in the sequence

        Returns:
            loss: absolute difference between hr_t and mean of hr_seq
        """
        ctx.hr_seq = hr_seq
        ctx.hr_mean = hr_seq.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        
        # Calculate absolute difference between current HR and mean HR
        loss = torch.abs(hr_t - ctx.hr_mean)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the smoothness loss.

        Args:
            ctx: context with saved tensors
            grad_output: gradient from upstream

        Returns:
            gradients for each input
        """
        hr_t, = ctx.saved_tensors
        hr_seq = ctx.hr_seq
        
        # Vectorized implementation
        mask = hr_seq != hr_t
        if mask.sum() > 0:
            # Sum gradients for all other elements in sequence
            output = (1 / ctx.T) * torch.sum(torch.sign(ctx.hr_mean - hr_seq[mask]))
            # Add gradient for current element
            output = output + (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t)
        else:
            # Handle case when hr_seq contains only hr_t
            output = (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t)
            
        # Scale by incoming gradient
        return output * grad_output, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        """
        Jacobian-vector product for PyTorch 2.0+ compatibility.
        
        Args:
            ctx: context with saved tensors
            grad_inputs: incoming gradients
            
        Returns:
            result of Jacobian-vector product
        """
        if len(grad_inputs) > 0 and grad_inputs[0] is not None:
            hr_t_grad = grad_inputs[0]
            hr_t, = ctx.saved_tensors
            return (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t) * hr_t_grad
        return None


class TotalLoss(nn.Module):
    """
    Combined loss function for heart rate prediction model.
    Combines L1 loss with a custom smoothness loss.
    """
    def __init__(self, lambda_val=100):
        """
        Initialize loss function.
        
        Args:
            lambda_val: weight for the smoothness loss term
        """
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambda_val = lambda_val
        self.gru_outputs_considered = None
        self.smooth_loss = SmoothLoss()

    def forward(self, resnet_pred, gru_pred, average_pred, y, average_hr):
        """
        Calculate the combined loss.
        
        Args:
            resnet_pred: predictions from ResNet
            gru_pred: predictions from GRU
            average_pred: average predictions
            y: ground truth labels
            average_hr: average heart rate ground truth
            
        Returns:
            tuple of (total_loss, l1_loss, smooth_loss)
        """
        # L1 loss for ResNet predictions
        l1_loss = self.l1_loss(resnet_pred, y)
        # Additional L1 loss for average predictions
        l1_loss = l1_loss + self.l1_loss(average_pred, average_hr)
        
        # Initialize smooth loss
        smooth_loss = torch.tensor(0.0, device=y.device)
        
        # For the temporal relationship modeling,
        # six adjacent estimated HRs are used to compute the L_smooth
        T = int(gru_pred.shape[0] // 6)
        
        # Process in chunks of 6
        for i in range(T):
            pred_seq = gru_pred[i * 6: (i + 1) * 6].flatten()
            chunk_loss = torch.tensor(0.0, device=y.device)
            
            for hr_t in pred_seq:
                chunk_loss = chunk_loss + self.smooth_loss.apply(hr_t, pred_seq, 6)
            
            smooth_loss = smooth_loss + chunk_loss / 6
        
        # Handle remaining sequence (if any)
        if gru_pred.shape[0] % 6:
            pred_seq = gru_pred[T * 6:].flatten()
            chunk_loss = torch.tensor(0.0, device=y.device)
            
            for hr_t in pred_seq:
                chunk_loss = chunk_loss + self.smooth_loss.apply(hr_t, pred_seq, len(pred_seq))
            
            smooth_loss = smooth_loss + chunk_loss / len(pred_seq)
        
        # Calculate total combined loss
        total_loss = l1_loss + self.lambda_val * smooth_loss
        
        return total_loss, l1_loss, smooth_loss
