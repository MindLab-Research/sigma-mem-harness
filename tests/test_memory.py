import torch

from sigma_mem_harness.memory import SigmaMemory


def test_sigma_memory_is_additive():
    a = SigmaMemory.from_features(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    b = SigmaMemory.from_features(torch.tensor([[2.0, 0.0]]))
    merged = a.add(b)
    assert merged.count.item() == 3
    torch.testing.assert_close(merged.cov, torch.diag(torch.tensor([5.0, 1.0])))


def test_sigma_memory_subtract_is_exact_inverse():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    whole = SigmaMemory.from_features(features)
    first = SigmaMemory.from_features(features[:1])
    rest = whole.subtract(first)
    torch.testing.assert_close(rest.cov, SigmaMemory.from_features(features[1:]).cov)
    assert rest.count.item() == 1
