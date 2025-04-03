from typing import Dict, Any
import logging

class TrainingCallback:
    """Base callback class for training events."""
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Called at the beginning of training."""
        pass
        
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training."""
        pass
        
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of an epoch."""
        pass
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of an epoch."""
        pass
        
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of a batch."""
        pass
        
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of a batch."""
        pass
