import pytest

from M_AD.engine.hooks.focus_adapter_only_hook import FocusAdapterOnlyHook


class DummyParam:

    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class DummyRunner:

    def __init__(self, model):
        self.model = model


class DummyBBoxHead:

    pass


class DummyFocusModel:

    def __init__(self, with_adapter=True):
        self.bbox_head = DummyBBoxHead()
        if with_adapter:
            self.bbox_head.focus_support_adapter = object()
        self._params = {
            "backbone.weight": DummyParam(),
            "neck.weight": DummyParam(),
            "bbox_head.cross_attention.weight": DummyParam(),
            "bbox_head.visual_fc.weight": DummyParam(),
            "bbox_head.rtm_cls.0.weight": DummyParam(),
            "bbox_head.rtm_cls_heads.0.log_scale_t1": DummyParam(),
            "bbox_head.rtm_reg.0.weight": DummyParam(),
            "bbox_head.rtm_ang.0.weight": DummyParam(),
        }
        if with_adapter:
            self._params.update({
                "bbox_head.focus_support_adapter.delta.0.weight": DummyParam(),
                "bbox_head.focus_support_adapter.delta.0.bias": DummyParam(),
                "bbox_head.focus_support_adapter.delta.2.weight": DummyParam(),
                "bbox_head.focus_support_adapter.delta.2.bias": DummyParam(),
            })

    def named_parameters(self):
        return self._params.items()


def test_focus_adapter_only_hook_allows_only_focus_support_adapter_params():
    model = DummyFocusModel()
    hook = FocusAdapterOnlyHook()

    hook.before_train(DummyRunner(model))

    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    assert trainable == {
        "bbox_head.focus_support_adapter.delta.0.weight",
        "bbox_head.focus_support_adapter.delta.0.bias",
        "bbox_head.focus_support_adapter.delta.2.weight",
        "bbox_head.focus_support_adapter.delta.2.bias",
    }


def test_focus_adapter_only_hook_fails_when_adapter_is_missing():
    model = DummyFocusModel(with_adapter=False)
    hook = FocusAdapterOnlyHook()

    with pytest.raises(RuntimeError, match="bbox_head.focus_support_adapter"):
        hook.before_train(DummyRunner(model))
