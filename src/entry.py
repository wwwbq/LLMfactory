from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llamafactory.extras.callbacks import LogCallback
from llamafactory.hparams import get_train_args
from llamafactory.trainer import TaskEngine

if TYPE_CHECKING:
    from transformers import TrainerCallback


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    callbacks.append(LogCallback(training_args.output_dir))
    
    engine_type = "run_" + finetuning_args.stage
    
    TaskEngine.get(engine_type)(
        model_args, data_args, training_args, finetuning_args, generating_args, callbacks
    )


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
