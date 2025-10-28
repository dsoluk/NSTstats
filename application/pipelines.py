# Temporary application-layer re-exports to keep backward compatibility
try:
    from pipelines import StatPipelineFactory, NSTPlayerPipeline  # type: ignore
except Exception:  # pragma: no cover
    class StatPipelineFactory:  # type: ignore
        @staticmethod
        def create(*args, **kwargs):
            raise ImportError("Original pipelines.StatPipelineFactory not available")
    class NSTPlayerPipeline:  # type: ignore
        pass
