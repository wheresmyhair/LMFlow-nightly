import subprocess
import sys
import unittest
from unittest.mock import patch

from lmflow.args import DatasetArguments, EvaluatorArguments, FinetunerArguments, InferencerArguments, ModelArguments
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.pipeline.evaluation import Evaluator
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer

MODEL_NAME = "gpt2"


class AutoPipelineTest(unittest.TestCase):
    def test_evaluator_compatibility_import(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "always::DeprecationWarning",
                "-c",
                "from lmflow.pipeline.evaluator import Evaluator; print(Evaluator.__module__)",
            ],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("lmflow.pipeline.evaluator is deprecated", completed.stderr)
        self.assertIn("lmflow.pipeline.evaluation.evaluator", completed.stdout)
        self.assertEqual(Evaluator.__module__, "lmflow.pipeline.evaluation.evaluator")

    def test_get_evaluator_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        evaluator_args = EvaluatorArguments()
        with patch.object(Evaluator, "__init__", return_value=None):
            pipeline = AutoPipeline.get_pipeline("evaluator", model_args, dataset_args, evaluator_args)

        self.assertTrue(isinstance(pipeline, Evaluator))

    def test_get_finetuner_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        finetuner_args = FinetunerArguments(output_dir="~/tmp")
        with patch.object(Finetuner, "__init__", return_value=None):
            pipeline = AutoPipeline.get_pipeline("finetuner", model_args, dataset_args, finetuner_args)

        self.assertTrue(isinstance(pipeline, Finetuner))

    def test_get_inferencer_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        inferencer_args = InferencerArguments()
        with patch.object(Inferencer, "__init__", return_value=None):
            pipeline = AutoPipeline.get_pipeline("inferencer", model_args, dataset_args, inferencer_args)

        self.assertTrue(isinstance(pipeline, Inferencer))

    def test_get_unsupported_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()

        with self.assertRaisesRegex(NotImplementedError, 'Pipeline "unsupported" is not supported'):
            pipeline = AutoPipeline.get_pipeline("unsupported", model_args, dataset_args, None)
