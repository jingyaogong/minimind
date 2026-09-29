"""Out-of-tree vLLM model registration for MiniMind."""


def register() -> None:
    from vllm import ModelRegistry

    architecture = "MiniMindForCausalLM"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture,
            "minimind_vllm.model:MiniMindForCausalLM",
        )
