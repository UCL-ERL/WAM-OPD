from wam_opd.cli import _doctor, build_parser


def test_cli_parser_exposes_doctor_and_manifest_validation() -> None:
    assert build_parser().parse_args(["doctor"]).command == "doctor"
    args = build_parser().parse_args(["manifest", "validate", "manifest.json"])
    assert args.manifest_command == "validate"


def test_doctor_reports_missing_runtime_sources() -> None:
    result = _doctor(require_gpu=False, require_models=False)
    assert result["schema"] == "wam_opd_doctor_v1"
    assert result["status"] == "FAIL"
    assert "lingbot_va_source" in result["errors"]
