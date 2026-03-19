"""
Advanced Evaluation Metrics for Geometric Algebra CNN Comparison

Implements metrics for:
- Predictive performance (Accuracy, F1-score, AUC-ROC)
- Parameter efficiency (parameter counts and ratios)
- Robustness to transformations (rotations, illumination changes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import torchvision.transforms as transforms
from tqdm import tqdm


class AdvancedMetricsTracker:
    """
    Tracks advanced evaluation metrics during training and testing.
    """
    
    def __init__(self, n_classes=9):
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.correct = 0
        self.total = 0
    
    def update(self, outputs, labels):
        """
        Update metrics with batch results.
        
        Args:
            outputs: Model output logits (B, n_classes)
            labels: True labels (B,)
        """
        # Get predicted class
        _, predicted = outputs.max(1)
        
        # Update accuracy
        self.correct += predicted.eq(labels).sum().item()
        self.total += labels.size(0)
        
        # Store predictions and labels (for F1, AUC computation)
        self.predictions.extend(predicted.cpu().numpy())
        self.true_labels.extend(labels.cpu().numpy())
        
        # Store probabilities (for AUC computation)
        probs = F.softmax(outputs, dim=1)
        self.probabilities.extend(probs.detach().cpu().numpy())
    
    def get_accuracy(self):
        """Return accuracy."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total
    
    def get_f1_score(self, average='weighted'):
        """
        Compute F1-score.
        
        Args:
            average: 'weighted' (default), 'macro', 'micro'
        
        Returns:
            F1-score (0-1)
        """
        if len(self.predictions) == 0:
            return 0.0
        
        return f1_score(
            self.true_labels,
            self.predictions,
            average=average,
            zero_division=0
        )
    
    def get_auc_score(self):
        """
        Compute AUC-ROC score (One-vs-Rest for multiclass).
        
        Returns:
            AUC-ROC score (0-1)
        """
        if len(self.probabilities) == 0 or len(np.unique(self.true_labels)) < 2:
            return 0.0
        
        try:
            # Binarize labels for multiclass AUC
            y_true_bin = label_binarize(
                self.true_labels,
                classes=range(self.n_classes)
            )
            
            # Compute AUC-ROC
            auc = roc_auc_score(
                y_true_bin,
                np.array(self.probabilities),
                multi_class='ovr',
                average='weighted'
            )
            return auc
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
            return 0.0
    
    def get_confusion_matrix(self):
        """Return confusion matrix."""
        if len(self.predictions) == 0:
            return None
        
        return confusion_matrix(
            self.true_labels,
            self.predictions,
            labels=range(self.n_classes)
        )
    
    def get_classification_report(self):
        """Return detailed classification report."""
        if len(self.predictions) == 0:
            return None
        
        return classification_report(
            self.true_labels,
            self.predictions,
            labels=range(self.n_classes),
            zero_division=0
        )
    
    def get_summary(self):
        """Return all metrics as dictionary."""
        return {
            'accuracy': self.get_accuracy(),
            'f1_weighted': self.get_f1_score('weighted'),
            'f1_macro': self.get_f1_score('macro'),
            'auc_roc': self.get_auc_score(),
            'total_samples': self.total
        }


class RobustnessEvaluator:
    """
    Evaluates model robustness to transformations.
    
    Tests performance under:
    - Rotations (±30 degrees)
    - Illumination changes (brightness/contrast)
    - Gaussian blur (σ=0.5 to 2.0)
    - Noise (Gaussian, Salt&Pepper)
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.results = {}
    
    def evaluate_rotation_robustness(self, dataloader, angles=[-30, -15, 0, 15, 30]):
        """
        Test robustness to image rotations.
        
        Args:
            dataloader: Test dataloader
            angles: List of rotation angles in degrees
        
        Returns:
            Dictionary with accuracy for each angle
        """
        results = {}
        
        for angle in angles:
            correct = 0
            total = 0
            
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(dataloader, desc=f'Rotation {angle}°', leave=False):
                    # Apply rotation
                    if angle != 0:
                        rotated = transforms.functional.rotate(images, angle)
                    else:
                        rotated = images
                    
                    images = rotated.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100.0 * correct / total
            results[f'{angle}°'] = accuracy
        
        self.results['rotation'] = results
        return results
    
    def evaluate_illumination_robustness(self, dataloader, 
                                        brightness_factors=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """
        Test robustness to illumination changes.
        
        Args:
            dataloader: Test dataloader
            brightness_factors: List of brightness multipliers
        
        Returns:
            Dictionary with accuracy for each brightness factor
        """
        results = {}
        
        for factor in brightness_factors:
            correct = 0
            total = 0
            
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(dataloader, 
                                          desc=f'Brightness {factor:.2f}x', 
                                          leave=False):
                    # Apply brightness adjustment
                    adjusted = images * factor
                    adjusted = torch.clamp(adjusted, 0, 1)
                    
                    images = adjusted.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100.0 * correct / total
            results[f'{factor:.2f}x'] = accuracy
        
        self.results['illumination'] = results
        return results
    
    def evaluate_blur_robustness(self, dataloader, kernel_sizes=[3, 5, 7, 9]):
        """
        Test robustness to Gaussian blur.
        
        Args:
            dataloader: Test dataloader
            kernel_sizes: List of blur kernel sizes
        
        Returns:
            Dictionary with accuracy for each kernel size
        """
        results = {}
        
        for kernel_size in kernel_sizes:
            correct = 0
            total = 0
            
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(dataloader, 
                                          desc=f'Blur kernel={kernel_size}', 
                                          leave=False):
                    # Apply Gaussian blur
                    blurred = transforms.GaussianBlur(
                        kernel_size=kernel_size,
                        sigma=(0.1, 2.0)
                    )(images)
                    
                    images = blurred.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100.0 * correct / total
            results[f'kernel={kernel_size}'] = accuracy
        
        self.results['blur'] = results
        return results
    
    def evaluate_noise_robustness(self, dataloader, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """
        Test robustness to Gaussian noise.
        
        Args:
            dataloader: Test dataloader
            noise_levels: List of noise standard deviations
        
        Returns:
            Dictionary with accuracy for each noise level
        """
        results = {}
        
        for noise_level in noise_levels:
            correct = 0
            total = 0
            
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(dataloader, 
                                          desc=f'Noise σ={noise_level:.2f}', 
                                          leave=False):
                    # Add Gaussian noise
                    noise = torch.randn_like(images) * noise_level
                    noisy = images + noise
                    noisy = torch.clamp(noisy, 0, 1)
                    
                    images = noisy.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100.0 * correct / total
            results[f'σ={noise_level:.2f}'] = accuracy
        
        self.results['noise'] = results
        return results
    
    def get_summary(self):
        """Return all robustness results."""
        return self.results


class ParameterEfficiencyAnalyzer:
    """
    Analyzes parameter efficiency of models.
    """
    
    def __init__(self):
        self.analysis = {}
    
    def count_parameters(self, model):
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_model_size(self, model):
        """Get model size in MB."""
        param_count = self.count_parameters(model)
        # Assuming 32-bit floats (4 bytes per parameter)
        size_mb = (param_count * 4) / (1024 * 1024)
        return size_mb
    
    def analyze(self, models_dict, accuracies_dict):
        """
        Analyze parameter efficiency of models.
        
        Args:
            models_dict: Dictionary of {model_name: model}
            accuracies_dict: Dictionary of {model_name: accuracy}
        
        Returns:
            Dictionary with efficiency metrics
        """
        results = {}
        
        for name, model in models_dict.items():
            param_count = self.count_parameters(model)
            accuracy = accuracies_dict.get(name, 0.0)
            
            # Compute efficiency metrics
            params_per_1pct = param_count / max(accuracy, 0.1)  # Parameters needed per 1% accuracy
            
            results[name] = {
                'parameters': param_count,
                'model_size_mb': self.get_model_size(model),
                'accuracy': accuracy,
                'params_per_1pct_acc': params_per_1pct
            }
        
        self.analysis = results
        return results
    
    def get_efficiency_comparison(self):
        """
        Compare efficiency across models.
        
        Returns:
            Sorted list of models by parameter efficiency
        """
        if not self.analysis:
            return None
        
        # Sort by params_per_1pct (lower is better)
        sorted_models = sorted(
            self.analysis.items(),
            key=lambda x: x[1]['params_per_1pct_acc']
        )
        
        return sorted_models


print("✓ Advanced metrics and robustness evaluation defined")
