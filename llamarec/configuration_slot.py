# coding=utf-8
from .configuration import LlamaRecConfig


class LlamaRecSlotConfig(LlamaRecConfig):
    model_type = "llama-rec-slot"

    def __init__(
        self,
        item_token_size=4,
        use_slot_input_embeddings=True,
        use_slot_prelogits_embeddings=True,
        use_slot_output_heads=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.item_token_size = item_token_size
        self.use_slot_input_embeddings = use_slot_input_embeddings
        self.use_slot_prelogits_embeddings = use_slot_prelogits_embeddings
        self.use_slot_output_heads = use_slot_output_heads


__all__ = ["LlamaRecSlotConfig"]
