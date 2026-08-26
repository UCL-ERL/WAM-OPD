PYTHON ?= python3

.PHONY: compile test test-runtime

compile:
	$(PYTHON) -m compileall -q experiments tests

test: compile
	$(PYTHON) -m pytest -q \
		tests/test_opd_task_specs.py \
		experiments/test_qualified_success_path_pipeline.py \
		experiments/test_scaled_qualified_success_path_pipeline.py \
		experiments/test_stage_h_task_progress.py

test-runtime:
	$(PYTHON) -m pytest -q \
		experiments/test_joint_lora.py \
		experiments/test_joint_lora_fp32.py \
		experiments/test_waopd_native_closed_loop_runner.py
