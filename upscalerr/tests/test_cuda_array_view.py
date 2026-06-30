import torch


def test_as_tensor_from_cai():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    src = torch.randint(0, 256, (8, 8, 4), dtype=torch.uint8, device=device)
    cai = src.__cuda_array_interface__
    view = torch.as_tensor(cai, device=device)
    assert view.data_ptr() == src.data_ptr()
    assert view.shape == (8, 8, 4)
