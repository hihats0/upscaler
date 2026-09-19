import torch

from upscaler.models.repair import RepairBgr255, RepairNet


def test_repair_starts_as_identity():
    net = RepairNet(16, 2)
    x = torch.rand(2, 3, 64, 96)
    assert torch.allclose(net(x), x)


def test_repair_bgr255_roundtrip_identity():
    wrap = RepairBgr255(RepairNet(16, 2))
    x = torch.randint(0, 256, (1, 3, 32, 32)).float()
    assert torch.equal(wrap(x), x)
