import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main


class FakeModelClass:
    loaded_model = object()

    @classmethod
    def create_from_arg_string(cls, model_args, runtime_args):
        del model_args, runtime_args
        return cls.loaded_model


class FakeTaskManager:
    def match_tasks(self, task_list):
        return task_list


class MainEvaluatorArgumentsTest(unittest.TestCase):
    def test_loaded_model_is_passed_separately_from_model_name(self):
        args = SimpleNamespace(
            tasks="mmmu",
            include_path=None,
            verbosity="INFO",
            model="internvl2",
            output_path=None,
            hf_hub_log_args="",
            log_samples=False,
            predict_only=False,
            fewshot_as_multiturn=False,
            apply_chat_template=False,
            limit=None,
            cache_requests=None,
            model_args="pretrained=/models/InternVL2-8B",
            batch_size="1",
            device=None,
            pseudo_quant=False,
            num_fewshot=None,
            max_batch_size=32,
            use_cache=None,
            check_integrity=False,
            write_out=False,
            system_instruction=None,
            gen_kwargs="",
            timezone="Asia/Singapore",
            seed=[0, 1234, 1234, 1234],
        )
        evaluate = Mock(return_value=None)

        with (
            patch.object(main, "TaskManager", return_value=FakeTaskManager()),
            patch.object(main, "EvaluationTracker", return_value=SimpleNamespace()),
            patch.object(main, "get_model", return_value=FakeModelClass),
            patch.object(main.evaluator, "simple_evaluate", evaluate),
        ):
            main.cli_evaluate_single(args)

        call_kwargs = evaluate.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "internvl2")
        self.assertIs(call_kwargs["lm"], FakeModelClass.loaded_model)


if __name__ == "__main__":
    unittest.main()
