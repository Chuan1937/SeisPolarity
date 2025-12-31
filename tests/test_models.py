import torch
print("Imported torch")
import pytest
from seispolarity.models import SCSN, PPNet, DitingMotion, EQPolarityCCT
print("Imported models")

def test_scsn():
    model = SCSN()
    x = torch.randn(2, 1, 400)
    y = model(x)
    assert y.shape == (2, 3)

def test_ppnet():
    model = PPNet(input_len=400, input_channels=1)
    x = torch.randn(2, 1, 400)
    y1, y2 = model(x)
    assert y1.shape == (2, 400, 1)
    assert y2.shape == (2, 3)

def test_diting_motion():
    model = DitingMotion(input_channels=2)
    x = torch.randn(2, 2, 128) # DiTingMotion seems to expect 128 length based on comments in original code?
    # Original code comment: # Keras 输入形状: (batch_size, length, channels) -> (N, 128, 2)
    # So length is 128.
    
    outputs = model(x)
    assert len(outputs) == 8
    for out in outputs:
        assert out.shape == (2, 3)

def test_eqpolarity():
    model = EQPolarityCCT(input_length=200)
    x = torch.randn(2, 1, 200)
    y = model(x)
    assert y.shape == (2, 1)

if __name__ == "__main__":
    print("Testing SCSN...")
    test_scsn()
    print("Testing PPNet...")
    test_ppnet()
    print("Testing DitingMotion...")
    test_diting_motion()
    print("Testing EQPolarity...")
    test_eqpolarity()
    print("All tests passed!")
