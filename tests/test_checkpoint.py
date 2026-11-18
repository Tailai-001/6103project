"""A failed NAS save must preserve the previous readable checkpoint."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from train import save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_interrupted_save_preserves_previous_checkpoint(self):
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'best.pth'
            save_checkpoint(path,model,optimizer,scheduler,1,.5,{'seed':42})
            original=path.read_bytes()
            def fail_save(state,target):
                Path(target).write_bytes(b'incomplete')
                raise OSError('simulated storage interruption')
            with patch('train.torch.save',side_effect=fail_save):
                with self.assertRaises(OSError):
                    save_checkpoint(path,model,optimizer,scheduler,2,.6,{'seed':42})
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(torch.load(path,weights_only=True)['epoch'],1)
            save_checkpoint(path,model,optimizer,scheduler,3,.7,{'seed':42})
            self.assertEqual(torch.load(path,weights_only=True)['epoch'],3)
            self.assertFalse(Path(str(path)+'.tmp').exists())
