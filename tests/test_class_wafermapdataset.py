import numpy as np
from src.pandas_data import WaferMapDataset

def test_wafermapdataset_getitem():
    # Fake 2D array mimicking a wafer map
    test_maps = [np.zeros((32, 32)), np.ones((32, 32))*127, np.random.rand(32, 32)*200]
    labels = [0, 1, 2]

    dataset = WaferMapDataset(test_maps, labels, transform=None)
    for i in range(len(dataset)):
        img, label = dataset[i]

        # assert img.shape[0] in (1, 3) or img.shape[-1] in (1, 3)   # Should be 3 channel
        # assert isinstance(label, int)

        # Handle PIL Image case
        if hasattr(img, "mode"):
            # Should be RGB, size tuple
            assert img.mode in ["RGB", "L"]
            assert isinstance(img.size, tuple)
        else:
            # If a Tensor (e.g., after ToTensor transform)
            assert hasattr(img, "shape")
            assert img.shape[0] in (1, 3) or img.shape[-1] in (1, 3)
        assert isinstance(label, int)