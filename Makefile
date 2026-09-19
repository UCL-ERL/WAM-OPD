PYTHON ?= python3

.PHONY: compile hygiene test test-runtime

compile:
	$(PYTHON) -m compileall -q experiments tests

hygiene:
	$(PYTHON) scripts/check_repository_hygiene.py

test: compile hygiene
	$(PYTHON) -m pytest -q \
		tests/test_repository_hygiene.py \
		tests/test_bootstrap_dependencies.py \
		tests/test_opd_task_specs.py \
		experiments/test_qualified_success_path_pipeline.py \
		experiments/test_scaled_qualified_success_path_pipeline.py \
		experiments/test_stage_h_task_progress.py

test-runtime:
	$(PYTHON) -m pytest -q \
		experiments/test_joint_lora.py \
		experiments/test_joint_lora_fp32.py \
		experiments/test_waopd_native_closed_loop_runner.py
