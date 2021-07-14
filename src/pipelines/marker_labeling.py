from src.pipelines.full_pipelines import Pipeline


class MarkerLabelingPipeline(Pipeline):
    def __init__(self, input_dir:str, output_dir:str):
        super().__init__(input_dir=input_dir, output_dir=output_dir)
