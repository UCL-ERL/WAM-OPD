"""Unit tests for frozen Stage H task-progress and promotion logic."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.stage_h_task_progress import (
    classify_behavior_pair,
    collect_task_progress,
)


class _Pose:
    def __init__(self, position, quaternion=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray(quaternion, dtype=np.float64)


class _Actor:
    def __init__(self, position, quaternion=(1.0, 0.0, 0.0, 0.0)):
        self._position = position
        self._quaternion = quaternion

    def get_pose(self):
        return _Pose(self._position, self._quaternion)


class _Cabinet:
    def __init__(self, target):
        self._target = target

    def get_functional_point(self, index):
        assert index == 0
        return np.asarray(self._target, dtype=np.float64)


class _Robot:
    def __init__(self, *, left_open=False, right_open=False):
        self.left_open = left_open
        self.right_open = right_open

    def is_left_gripper_open(self):
        return self.left_open

    def is_right_gripper_open(self):
        return self.right_open


class _CabinetEnv:
    def __init__(self, position, *, open_gripper=False):
        self.object = _Actor(position)
        self.cabinet = _Cabinet([0.0, 0.0, 0.3])
        self.origin_z = 0.2
        self.arm_tag = "left"
        self.robot = _Robot(left_open=open_gripper)


class _BottlesEnv:
    def __init__(self, reward):
        self.reward = reward

    def stage_reward(self):
        return self.reward


class _FunctionalActor(_Actor):
    def get_functional_point(self, index, kind):
        assert kind == "pose"
        return _Pose(self._position)


class _HandoverEnv:
    def __init__(self, box_position, *, right_open=False):
        self.box = _FunctionalActor(box_position)
        self.target_box = _FunctionalActor([0.2, 0.1, 0.75])
        self.robot = _Robot(right_open=right_open)

    def is_right_gripper_open(self):
        return self.robot.is_right_gripper_open()


class _Microphone:
    def __init__(self, position):
        self._position = np.asarray(position, dtype=np.float64)

    def get_functional_point(self, index):
        assert index == 0
        return self._position


class _HandoverMicEnv:
    def __init__(
        self,
        position,
        *,
        contact=False,
        receiver_closed=False,
        giver_open=False,
    ):
        self.microphone = _Microphone(position)
        self.grasp_arm_tag = "left"
        self.handover_arm_tag = "right"
        self._contact = contact
        self._receiver_closed = receiver_closed
        self._giver_open = giver_open

    def get_gripper_actor_contact_position(self, actor_name):
        assert actor_name == "018_microphone"
        return [[0.0, 0.0, 0.0]] if self._contact else []

    def is_right_gripper_close(self):
        return self._receiver_closed

    def is_left_gripper_open(self):
        return self._giver_open


class _ClickBellEnv:
    def __init__(self, stage_success):
        self.stage_success_tag = stage_success


class _BlocksEnv:
    def __init__(self, positions, *, left_open=False, right_open=False):
        self.block1 = _Actor(positions[0])
        self.block2 = _Actor(positions[1])
        self.block3 = _Actor(positions[2])
        self._left_open = left_open
        self._right_open = right_open

    def is_left_gripper_open(self):
        return self._left_open

    def is_right_gripper_open(self):
        return self._right_open


class _A2BLeftEnv:
    def __init__(
        self,
        object_position,
        target_position=(0.0, 0.0, 0.75),
        *,
        left_open=False,
        right_open=False,
    ):
        self.object = _Actor(object_position)
        self.target_object = _Actor(target_position)
        self.robot = _Robot(left_open=left_open, right_open=right_open)


class _StampSealEnv:
    def __init__(
        self,
        seal_position,
        target_position=(0.0, 0.0, 0.75),
        *,
        left_open=False,
        right_open=False,
    ):
        self.seal = _Actor(seal_position)
        self.target = _Actor(target_position)
        self.robot = _Robot(left_open=left_open, right_open=right_open)


class _BreadBasketEnv:
    def __init__(self, bread_positions, *, left_open=False, right_open=False):
        self.breadbasket = _Actor([0.0, 0.0, 0.73])
        self.bread = [_Actor(position) for position in bread_positions]
        self.table_z_bias = 0.0
        self.robot = _Robot(left_open=left_open, right_open=right_open)


class _PlasticBox(_Actor):
    def get_functional_point(self, index):
        return np.asarray(
            ([-0.02, 0.0, 0.75], [0.02, 0.0, 0.75])[index],
            dtype=np.float64,
        )


class _CansEnv:
    def __init__(self, positions, *, left_open=False, right_open=False):
        self.plasticbox = _PlasticBox([0.0, 0.0, 0.75])
        self.object1 = _Actor(positions[0])
        self.object2 = _Actor(positions[1])
        self._left_open = left_open
        self._right_open = right_open

    def is_left_gripper_open(self):
        return self._left_open

    def is_right_gripper_open(self):
        return self._right_open


class _QRCodeEnv:
    def __init__(
        self,
        position,
        quaternion,
        *,
        left_open=False,
        right_open=False,
    ):
        self.qrcode = _Actor(position, quaternion)
        self.table_z_bias = 0.0
        self._left_open = left_open
        self._right_open = right_open

    def is_left_gripper_open(self):
        return self._left_open

    def is_right_gripper_open(self):
        return self._right_open


class _Switch:
    def __init__(self, qpos, limits=(0.0, 1.0)):
        self._qpos = qpos
        self._limits = limits

    def get_qpos(self):
        return np.asarray([self._qpos], dtype=np.float64)

    def get_qlimits(self):
        return np.asarray([self._limits], dtype=np.float64)


class _TurnSwitchEnv:
    def __init__(self, qpos):
        self.switch = _Switch(qpos)


class _MicrowaveEnv:
    def __init__(self, qpos, limits=(0.0, 1.0)):
        self.microwave = _Switch(qpos, limits=limits)


class _StaplerPadEnv:
    def __init__(
        self,
        position,
        *,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        left_open=False,
        right_open=False,
    ):
        self.stapler = _Actor(position, quaternion)
        self.pad = _Actor([0.0, 0.0, 0.3])
        self.robot = _Robot(left_open=left_open, right_open=right_open)


class _ShoeEnv:
    def __init__(
        self,
        position,
        *,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        left_open=False,
        right_open=False,
    ):
        self.shoe = _Actor(position, quaternion)
        self._left_open = left_open
        self._right_open = right_open

    def is_left_gripper_open(self):
        return self._left_open

    def is_right_gripper_open(self):
        return self._right_open


class _Scanner(_Actor):
    def __init__(self, functional_pose):
        self._functional_pose = np.asarray(
            functional_pose, dtype=np.float64
        )

    def get_functional_point(self, index):
        assert index == 0
        return self._functional_pose.copy()


class _ScanObjectEnv:
    def __init__(
        self,
        object_position,
        *,
        scanner_functional_pose=(0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
        left_closed=False,
        right_closed=False,
    ):
        self.object = _Actor(object_position)
        self.scanner = _Scanner(scanner_functional_pose)
        self._left_closed = left_closed
        self._right_closed = right_closed

    def is_left_gripper_close(self):
        return self._left_closed

    def is_right_gripper_close(self):
        return self._right_closed


class StageHTaskProgressTest(unittest.TestCase):
    def test_new_candidate_progress_matches_official_predicates(self):
        right = _A2BLeftEnv(
            [0.13, 0.0, 0.75], left_open=True, right_open=True
        )
        self.assertEqual(
            collect_task_progress("place_a2b_right", right)["ordinal_stage"],
            3,
        )

        bread = _BreadBasketEnv(
            [[0.01, 0.01, 0.74], [-0.01, -0.01, 0.74]],
            left_open=True,
            right_open=True,
        )
        bread_progress = collect_task_progress("place_bread_basket", bread)
        self.assertEqual(bread_progress["placed_bread_count"], 2)
        self.assertEqual(bread_progress["ordinal_stage"], 3)

        cans = _CansEnv(
            [[-0.02, 0.0, 0.75], [0.02, 0.0, 0.75]],
            left_open=True,
            right_open=True,
        )
        cans_progress = collect_task_progress("place_cans_plasticbox", cans)
        self.assertEqual(cans_progress["placed_can_count"], 2)
        self.assertEqual(cans_progress["ordinal_stage"], 3)

        qrcode = _QRCodeEnv(
            [0.2, -0.15, 0.74],
            [-0.707, -0.707, 0.0, 0.0],
            left_open=True,
            right_open=True,
        )
        qr_progress = collect_task_progress("rotate_qrcode", qrcode)
        self.assertTrue(qr_progress["quaternion_valid"])
        self.assertEqual(qr_progress["ordinal_stage"], 3)

    def test_stamp_seal_matches_official_xy_and_release_predicate(self):
        far = collect_task_progress(
            "stamp_seal", _StampSealEnv([0.02, 0.02, 0.75])
        )
        one_axis = collect_task_progress(
            "stamp_seal", _StampSealEnv([0.009, 0.02, 0.75])
        )
        placed = collect_task_progress(
            "stamp_seal", _StampSealEnv([0.009, 0.009, 0.75])
        )
        released = collect_task_progress(
            "stamp_seal",
            _StampSealEnv(
                [0.009, 0.009, 0.75], left_open=True, right_open=True
            ),
        )
        strict_boundary = collect_task_progress(
            "stamp_seal",
            _StampSealEnv(
                [0.01, 0.009, 0.75], left_open=True, right_open=True
            ),
        )
        self.assertEqual(
            [
                far["ordinal_stage"],
                one_axis["ordinal_stage"],
                placed["ordinal_stage"],
                released["ordinal_stage"],
                strict_boundary["ordinal_stage"],
            ],
            [0, 1, 2, 3, 1],
        )
        self.assertTrue(released["official_success"])
        self.assertFalse(strict_boundary["official_success"])

    def test_cabinet_milestones_match_conjunctive_success_predicate(self):
        self.assertEqual(
            collect_task_progress(
                "put_object_cabinet", _CabinetEnv([0.2, 0.0, 0.2])
            )["ordinal_stage"],
            0,
        )
        self.assertEqual(
            collect_task_progress(
                "put_object_cabinet", _CabinetEnv([0.2, 0.0, 0.3])
            )["ordinal_stage"],
            1,
        )
        self.assertEqual(
            collect_task_progress(
                "put_object_cabinet", _CabinetEnv([0.01, 0.01, 0.3])
            )["ordinal_stage"],
            2,
        )
        self.assertEqual(
            collect_task_progress(
                "put_object_cabinet",
                _CabinetEnv([0.01, 0.01, 0.3], open_gripper=True),
            )["ordinal_stage"],
            3,
        )

    def test_bottles_uses_official_stage_reward(self):
        progress = collect_task_progress(
            "put_bottles_dustbin", _BottlesEnv(2.0 / 3.0)
        )
        self.assertEqual(progress["ordinal_stage"], 2)
        self.assertAlmostEqual(progress["stage_reward"], 2.0 / 3.0)

    def test_handover_milestones_match_official_success_conjuncts(self):
        far = collect_task_progress(
            "handover_block", _HandoverEnv([0.0, 0.0, 0.85])
        )
        xy_aligned = collect_task_progress(
            "handover_block", _HandoverEnv([0.21, 0.11, 0.85])
        )
        placed = collect_task_progress(
            "handover_block", _HandoverEnv([0.21, 0.11, 0.755])
        )
        released = collect_task_progress(
            "handover_block",
            _HandoverEnv([0.21, 0.11, 0.755], right_open=True),
        )
        self.assertEqual(
            [far["ordinal_stage"], xy_aligned["ordinal_stage"], placed["ordinal_stage"], released["ordinal_stage"]],
            [0, 1, 2, 3],
        )

    def test_handover_mic_milestones_match_official_success_conjuncts(self):
        far = collect_task_progress(
            "handover_mic", _HandoverMicEnv([-0.1, 0.0, 0.8])
        )
        positioned = collect_task_progress(
            "handover_mic", _HandoverMicEnv([0.1, 0.0, 0.95])
        )
        received = collect_task_progress(
            "handover_mic",
            _HandoverMicEnv(
                [0.1, 0.0, 0.95], contact=True, receiver_closed=True
            ),
        )
        success = collect_task_progress(
            "handover_mic",
            _HandoverMicEnv(
                [0.1, 0.0, 0.95],
                contact=True,
                receiver_closed=True,
                giver_open=True,
            ),
        )

        self.assertEqual(
            [
                far["ordinal_stage"],
                positioned["ordinal_stage"],
                received["ordinal_stage"],
                success["ordinal_stage"],
            ],
            [0, 1, 2, 3],
        )
        self.assertTrue(success["official_success"])

    def test_place_shoe_matches_official_pose_release_predicate(self):
        target_quaternion = (-0.5, -0.5, 0.5, 0.5)
        far = collect_task_progress(
            "place_shoe", _ShoeEnv([0.1, -0.08, -3.0])
        )
        xy_only = collect_task_progress(
            "place_shoe", _ShoeEnv([0.0, -0.08, 9.0])
        )
        posed = collect_task_progress(
            "place_shoe",
            _ShoeEnv(
                [0.0, -0.08, -3.0], quaternion=target_quaternion
            ),
        )
        released = collect_task_progress(
            "place_shoe",
            _ShoeEnv(
                [0.0, -0.08, 9.0],
                quaternion=target_quaternion,
                left_open=True,
                right_open=True,
            ),
        )
        strict_xy_boundary = collect_task_progress(
            "place_shoe",
            _ShoeEnv(
                [0.05, -0.08, 9.0],
                quaternion=target_quaternion,
                left_open=True,
                right_open=True,
            ),
        )
        strict_quaternion_boundary = collect_task_progress(
            "place_shoe",
            _ShoeEnv(
                [0.0, -0.08, 9.0],
                quaternion=(0.43, 0.5, -0.5, -0.5),
                left_open=True,
                right_open=True,
            ),
        )

        self.assertEqual(
            [
                far["ordinal_stage"],
                xy_only["ordinal_stage"],
                posed["ordinal_stage"],
                released["ordinal_stage"],
            ],
            [0, 1, 2, 3],
        )
        self.assertTrue(released["official_success"])
        self.assertFalse(strict_xy_boundary["xy_valid"])
        self.assertFalse(strict_xy_boundary["official_success"])
        self.assertFalse(strict_quaternion_boundary["quaternion_valid"])
        self.assertFalse(strict_quaternion_boundary["official_success"])
        self.assertEqual(posed["shoe_z"], -3.0)
        self.assertEqual(released["shoe_z"], 9.0)
        self.assertFalse(released["shoe_z_is_official"])

    def test_scan_object_matches_official_projected_geometry_predicate(self):
        far = collect_task_progress(
            "scan_object", _ScanObjectEnv([0.03, 0.0, 0.55])
        )
        aligned_wrong_depth = collect_task_progress(
            "scan_object", _ScanObjectEnv([0.0, 0.0, 0.49])
        )
        geometry = collect_task_progress(
            "scan_object", _ScanObjectEnv([0.01, 0.02, 0.55])
        )
        success = collect_task_progress(
            "scan_object",
            _ScanObjectEnv(
                [0.01, 0.02, 0.55],
                left_closed=True,
                right_closed=True,
            ),
        )

        self.assertEqual(
            [
                far["ordinal_stage"],
                aligned_wrong_depth["ordinal_stage"],
                geometry["ordinal_stage"],
                success["ordinal_stage"],
            ],
            [0, 1, 2, 3],
        )
        self.assertAlmostEqual(success["scanner_axis_depth"], 0.05)
        self.assertAlmostEqual(success["projected_xyz_linf_error"], 0.02)
        self.assertTrue(success["official_success"])

    def test_scan_object_uses_scanner_orientation_and_strict_boundaries(self):
        # 180 degrees around y rotates scanner-local -z onto world +z.
        rotated_scanner = (0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0)
        rotated_success = collect_task_progress(
            "scan_object",
            _ScanObjectEnv(
                [0.0, 0.0, 0.45],
                scanner_functional_pose=rotated_scanner,
                left_closed=True,
                right_closed=True,
            ),
        )
        projection_boundary = collect_task_progress(
            "scan_object", _ScanObjectEnv([0.025, 0.0, 0.55])
        )
        zero_depth = collect_task_progress(
            "scan_object", _ScanObjectEnv([0.0, 0.0, 0.5])
        )
        upper_depth = collect_task_progress(
            "scan_object", _ScanObjectEnv([0.0, 0.0, 0.570001])
        )

        self.assertTrue(rotated_success["official_success"])
        self.assertAlmostEqual(rotated_success["scanner_axis_depth"], 0.05)
        self.assertFalse(projection_boundary["projected_xyz_valid"])
        self.assertFalse(zero_depth["depth_valid"])
        self.assertFalse(upper_depth["depth_valid"])

    def test_click_bell_uses_latched_official_success(self):
        self.assertEqual(
            collect_task_progress("click_bell", _ClickBellEnv(False))["ordinal_stage"],
            0,
        )
        self.assertEqual(
            collect_task_progress("click_bell", _ClickBellEnv(True))["ordinal_stage"],
            1,
        )

    def test_turn_switch_uses_official_joint_limit_threshold(self):
        below = collect_task_progress("turn_switch", _TurnSwitchEnv(0.949))
        reached = collect_task_progress("turn_switch", _TurnSwitchEnv(0.95))
        self.assertEqual(below["ordinal_stage"], 0)
        self.assertEqual(reached["ordinal_stage"], 1)
        self.assertFalse(below["switch_activated"])
        self.assertTrue(reached["switch_activated"])
        self.assertAlmostEqual(reached["normalized_progress"], 0.95)

    def test_open_microwave_uses_official_upper_limit_fraction(self):
        below = collect_task_progress(
            "open_microwave", _MicrowaveEnv(0.719, limits=(0.1, 1.2))
        )
        reached = collect_task_progress(
            "open_microwave", _MicrowaveEnv(0.72, limits=(0.1, 1.2))
        )
        self.assertEqual(below["ordinal_stage"], 0)
        self.assertEqual(reached["ordinal_stage"], 1)
        self.assertFalse(below["door_open"])
        self.assertTrue(reached["door_open"])
        self.assertAlmostEqual(reached["success_threshold"], 0.72)
        self.assertAlmostEqual(reached["normalized_progress"], 0.62 / 1.1)

    def test_move_stapler_pad_milestones_match_official_success_conjuncts(self):
        aligned_quaternion = (0.5, 0.5, 0.5, 0.5)
        far = collect_task_progress(
            "move_stapler_pad", _StaplerPadEnv([0.1, 0.0, 0.3])
        )
        positioned = collect_task_progress(
            "move_stapler_pad", _StaplerPadEnv([0.01, 0.01, 0.305])
        )
        aligned = collect_task_progress(
            "move_stapler_pad",
            _StaplerPadEnv(
                [0.01, 0.01, 0.305], quaternion=aligned_quaternion
            ),
        )
        released = collect_task_progress(
            "move_stapler_pad",
            _StaplerPadEnv(
                [0.01, 0.01, 0.305],
                quaternion=aligned_quaternion,
                left_open=True,
                right_open=True,
            ),
        )

        self.assertEqual(
            [
                far["ordinal_stage"],
                positioned["ordinal_stage"],
                aligned["ordinal_stage"],
                released["ordinal_stage"],
            ],
            [0, 1, 2, 3],
        )
        self.assertTrue(released["official_success"])

    def test_blocks_milestones_count_official_ordering_conjuncts(self):
        unordered = _BlocksEnv(
            [[0.2, 0.0, 0.75], [-0.2, 0.0, 0.75], [0.3, 0.0, 0.75]]
        )
        one_pair = _BlocksEnv(
            [[-0.10, -0.15, 0.75], [0.01, -0.15, 0.75], [0.3, 0.0, 0.75]]
        )
        both_pairs = _BlocksEnv(
            [[-0.10, -0.15, 0.75], [0.01, -0.15, 0.75], [0.09, -0.15, 0.75]]
        )
        released = _BlocksEnv(
            [[-0.10, -0.15, 0.75], [0.01, -0.15, 0.75], [0.09, -0.15, 0.75]],
            left_open=True,
            right_open=True,
        )
        self.assertEqual(
            [
                collect_task_progress("blocks_ranking_size", env)["ordinal_stage"]
                for env in (unordered, one_pair, both_pairs, released)
            ],
            [0, 1, 2, 3],
        )

    def test_place_a2b_left_matches_official_annulus_alignment_and_release(self):
        misaligned = _A2BLeftEnv([-0.1, 0.06, 0.75])
        aligned_outside_annulus = _A2BLeftEnv([-0.21, 0.0, 0.75])
        placed = _A2BLeftEnv([-0.1, 0.0, 0.75])
        released = _A2BLeftEnv(
            [-0.1, 0.0, 0.75], left_open=True, right_open=True
        )
        self.assertEqual(
            [
                collect_task_progress("place_a2b_left", env)["ordinal_stage"]
                for env in (
                    misaligned,
                    aligned_outside_annulus,
                    placed,
                    released,
                )
            ],
            [0, 1, 2, 3],
        )
        for boundary in (-0.08, -0.2):
            with self.subTest(boundary=boundary):
                progress = collect_task_progress(
                    "place_a2b_left",
                    _A2BLeftEnv(
                        [boundary, 0.0, 0.75],
                        left_open=True,
                        right_open=True,
                    ),
                )
                self.assertFalse(progress["distance_valid"])
                self.assertFalse(progress["official_success"])

    def test_stack_blocks_three_matches_official_vertical_pairs_and_release(self):
        unstacked = _BlocksEnv(
            [[0.0, 0.0, 0.75], [0.1, 0.0, 0.80], [0.2, 0.0, 0.85]]
        )
        one_pair = _BlocksEnv(
            [[0.0, 0.0, 0.75], [0.0, 0.0, 0.80], [0.2, 0.0, 0.85]]
        )
        stacked = _BlocksEnv(
            [[0.0, 0.0, 0.75], [0.0, 0.0, 0.80], [0.0, 0.0, 0.85]]
        )
        released = _BlocksEnv(
            [[0.0, 0.0, 0.75], [0.0, 0.0, 0.80], [0.0, 0.0, 0.85]],
            left_open=True,
            right_open=True,
        )
        self.assertEqual(
            [
                collect_task_progress("stack_blocks_three", env)[
                    "ordinal_stage"
                ]
                for env in (unstacked, one_pair, stacked, released)
            ],
            [0, 1, 2, 3],
        )

    def test_progress_improvement_promotes_only_an_ordinal_change(self):
        student = {
            "success": False,
            "completion_step": 20,
            "progress": {"ordinal_stage": 1, "xy_linf_error": 0.01},
        }
        bridge = {
            "success": False,
            "completion_step": 20,
            "progress": {"ordinal_stage": 2, "xy_linf_error": 0.04},
        }
        self.assertEqual(
            classify_behavior_pair(student, bridge), "positive_progress"
        )
        bridge["progress"]["ordinal_stage"] = 1
        self.assertEqual(
            classify_behavior_pair(student, bridge), "neutral_equal_progress"
        )

    def test_success_regression_dominates_progress(self):
        student = {
            "success": True,
            "completion_step": 10,
            "progress": {"ordinal_stage": 3},
        }
        bridge = {
            "success": False,
            "completion_step": 8,
            "progress": {"ordinal_stage": 3},
        }
        self.assertEqual(
            classify_behavior_pair(student, bridge), "negative_success"
        )


if __name__ == "__main__":
    unittest.main()
