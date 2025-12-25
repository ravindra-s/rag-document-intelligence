from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
import torch
import logging

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Unified local LLM wrapper supporting:
    - Seq2Seq models (T5 / FLAN-T5 / Long-T5)
    - Causal models (Qwen / Phi / Mistral)

    Handles prompt slicing correctly for causal models.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        logger.info("Initializing LocalLLM with model: %s", model_name)

        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.config.is_encoder_decoder:
            self.model_type = "seq2seq"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.model_type = "causal"
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval()

        if (
            getattr(self.tokenizer, "model_max_length", None) is None
            or self.tokenizer.model_max_length > 100_000
        ):
            self.tokenizer.model_max_length = 4096
            logger.warning(
                "Tokenizer had no valid model_max_length; defaulting to %d",
                self.tokenizer.model_max_length,
            )

        logger.info(
            "LocalLLM ready | model=%s | type=%s",
            model_name,
            self.model_type,
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        output_ids = outputs[0]

        # 🔑 Critical difference
        if self.model_type == "causal":
            generated_ids = output_ids[prompt_len:]
        else:
            generated_ids = output_ids

        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return decoded