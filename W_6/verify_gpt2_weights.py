"""Copy GPT-2 weights from HuggingFace and verify embedding parity."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from custom_gpt2 import GPT2Config, GPT2LMHeadModel


def to_custom_config(hf_config: AutoConfig) -> GPT2Config:
    """Translate HuggingFace GPT-2 config into our lightweight config."""
    cfg_dict = hf_config.to_dict()
    return GPT2Config(
        vocab_size=cfg_dict.get("vocab_size"),
        n_positions=cfg_dict.get("n_positions", cfg_dict.get("n_ctx")),
        n_ctx=cfg_dict.get("n_ctx", cfg_dict.get("n_positions")),
        n_embd=cfg_dict.get("n_embd"),
        n_layer=cfg_dict.get("n_layer"),
        n_head=cfg_dict.get("n_head"),
        n_inner=cfg_dict.get("n_inner"),
        activation_function=cfg_dict.get("activation_function", "gelu_new"),
        resid_pdrop=cfg_dict.get("resid_pdrop", 0.1),
        embd_pdrop=cfg_dict.get("embd_pdrop", 0.1),
        attn_pdrop=cfg_dict.get("attn_pdrop", 0.1),
        layer_norm_epsilon=cfg_dict.get("layer_norm_epsilon", 1e-5),
        initializer_range=cfg_dict.get("initializer_range", 0.02),
        use_cache=cfg_dict.get("use_cache", True),
        bos_token_id=cfg_dict.get("bos_token_id", 50256),
        eos_token_id=cfg_dict.get("eos_token_id", 50256),
    )


def main(model_name: str = "gpt2") -> None:
    """Load HF GPT-2 weights into the custom module and compare embeddings."""
    # 1) Load reference GPT-2 weights from HuggingFace.
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.eval()

    # 2) Instantiate the lightweight decoder with a matching config.
    custom_cfg = to_custom_config(hf_model.config)
    custom_model = GPT2LMHeadModel(custom_cfg)

    # 3) Copy the state dict; any mismatch means the layout diverges.
    missing, unexpected = custom_model.load_state_dict(hf_model.state_dict(), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")

    with torch.no_grad():
        # 4) Compare token and positional embedding tensors element-wise.
        token_diff = (custom_model.transformer.wte.weight - hf_model.transformer.wte.weight).abs().max()
        pos_diff = (custom_model.transformer.wpe.weight - hf_model.transformer.wpe.weight).abs().max()

    print(f"Max token embedding diff: {token_diff.item():.6e}")
    print(f"Max position embedding diff: {pos_diff.item():.6e}")

    if token_diff > 1e-6 or pos_diff > 1e-6:
        raise AssertionError("Embedding weights do not match within tolerance.")

    # A zero (or near-zero) max difference confirms exact copy.
    print("Embedding weights match within tolerance.")


if __name__ == "__main__":
    main()
