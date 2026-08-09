import logging
from copy import deepcopy

from lmflow.datasets import Dataset
from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.utils.lisa_trainer import DynamicLayerActivationCallback
from lmflow.pipeline.utils.rm_dataprocessor import RewardDataCollatorWithPadding
from lmflow.pipeline.utils.rm_trainer import RewardTrainer, compute_metrics

logger = logging.getLogger(__name__)


class RewardModelTuner(Finetuner):
    """Initializes the `RewardModelTuner` class.

    Parameters
    ----------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : RewardModelTunerArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.
    """

    def __init__(self, model_args, data_args, finetuner_args, *args, **kwargs):
        super().__init__(model_args, data_args, finetuner_args, *args, **kwargs)

    def tune(
        self, model: HFTextRegressionModel, dataset, transform_dataset_in_place=True, data_collator=None, **kwargs
    ):
        # 0. basic init
        if not transform_dataset_in_place:
            dataset = deepcopy(dataset)

        # 1. prepare dataset
        with self.finetuner_args.main_process_first(desc="dataset map tokenization"):
            tokenized_dataset = model.tokenize(dataset)
            if self.data_args.disable_group_texts:
                lm_dataset = tokenized_dataset
            else:
                lm_dataset = self.group_text(
                    tokenized_dataset,
                    model_max_length=model.get_max_length(),
                )
        train_dataset = lm_dataset.get_backend_dataset()
        logger.info(f"Number of train samples: {len(train_dataset)}")

        if self.finetuner_args.do_train and self.data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if self.finetuner_args.do_eval:
            eval_dataset_args = deepcopy(self.data_args)
            eval_dataset_args.dataset_path = self.finetuner_args.eval_dataset_path
            eval_dataset = Dataset(eval_dataset_args)
            with self.finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(eval_dataset)
                if self.data_args.disable_group_texts:
                    lm_dataset = tokenized_dataset
                else:
                    lm_dataset = self.group_text(
                        tokenized_dataset,
                        model_max_length=model.get_max_length(),
                    )
            eval_dataset = lm_dataset.get_backend_dataset()
            logger.info(f"Number of eval samples: {len(eval_dataset)}")

        if data_collator is None:
            data_collator = RewardDataCollatorWithPadding(
                tokenizer=model.get_tokenizer(), max_length=self.model_args.model_max_length
            )

        # 2. prepare trainer
        RewardModelingTrainer = RewardTrainer
        trainer_callbacks = []

        if self.finetuner_args.use_lisa:
            dynamic_layer_activation_callback = DynamicLayerActivationCallback(
                n_layers=self.finetuner_args.lisa_activated_layers,  # Number of layers to activate
                interval_steps=self.finetuner_args.lisa_interval_steps,  # Step interval to update active layers
                model=model.get_backend_model(),
                lisa_layers_attribute=self.finetuner_args.lisa_layers_attribute,
            )

            trainer_callbacks.append(dynamic_layer_activation_callback)

        trainer = RewardModelingTrainer(
            model=model.get_backend_model(),
            args=self.finetuner_args,
            train_dataset=train_dataset if self.finetuner_args.do_train else None,
            eval_dataset=eval_dataset if self.finetuner_args.do_eval else None,
            tokenizer=model.get_tokenizer(),
            data_collator=data_collator,
            compute_metrics=compute_metrics if self.finetuner_args.do_eval else None,
            callbacks=trainer_callbacks,
        )

        # 3. training
        if self.finetuner_args.do_train:
            checkpoint = None
            last_checkpoint = self.last_checkpoint
            if self.finetuner_args.resume_from_checkpoint is not None:
                checkpoint = self.finetuner_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

            if self.finetuner_args.gradient_checkpointing:
                if model.get_backend_model().config.use_cache:
                    logger.warning(
                        "Backend model config `use_cache=True` is incompatible with gradient checkpointing. "
                        "Setting `use_cache=False`."
                    )
                    model.get_backend_model().config.use_cache = False

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics

            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        kwargs = {"finetuned_from": self.model_args.model_name_or_path, "tasks": "reward-modeling"}
        if self.data_args.dataset_name is not None:
            kwargs["dataset_tags"] = self.data_args.dataset_name
            if self.data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = self.data_args.dataset_config_name
                kwargs["dataset"] = f"{self.data_args.dataset_name} {self.data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = self.data_args.dataset_name

        if self.finetuner_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

        return model
