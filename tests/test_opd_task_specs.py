import unittest

from experiments import opd_task_specs as MODULE


class TaskSpecTest(unittest.TestCase):
    def test_known_task_horizons(self) -> None:
        self.assertEqual(MODULE.resolve_task_chunks("move_stapler_pad"), 13)
        self.assertEqual(MODULE.resolve_task_chunks("open_microwave"), 48)
        self.assertEqual(MODULE.resolve_task_chunks("place_fan"), 13)
        self.assertEqual(MODULE.resolve_task_chunks("put_object_cabinet"), 23)
        self.assertEqual(MODULE.resolve_task_chunks("put_bottles_dustbin"), 54)
        self.assertEqual(MODULE.resolve_task_chunks("handover_mic"), 20)
        self.assertEqual(MODULE.resolve_task_chunks("place_shoe"), 17)
        self.assertEqual(MODULE.resolve_task_chunks("scan_object"), 17)

    def test_training_domain_is_easy_only(self) -> None:
        self.assertEqual(
            MODULE.require_training_task_config("demo_clean"),
            "demo_clean",
        )
        with self.assertRaisesRegex(ValueError, "demo_randomized only"):
            MODULE.require_training_task_config("demo_randomized")

    def test_override_supports_unregistered_tasks(self) -> None:
        self.assertEqual(MODULE.resolve_task_chunks("future_task", 31), 31)

    def test_unregistered_task_requires_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass --chunks explicitly"):
            MODULE.resolve_task_chunks("future_task")


if __name__ == "__main__":
    unittest.main()
