from upscalerr.pipeline.frame_state import FrameState, PresentSlot


def test_present_sequence():
    fs = FrameState()
    fs.enqueue_mid_then_real()
    assert fs.pop_present() == PresentSlot.MID
    assert fs.pop_present() == PresentSlot.REAL
    assert fs.pop_present() is None
