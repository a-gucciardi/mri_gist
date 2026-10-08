import asyncio
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ants
from click.testing import CliRunner
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from mri_gist.cli import cli
from mri_gist.conversion.formats import convert_format
from mri_gist.fractal.analysis import (
    DEFAULT_RESULT_COLUMNS,
    compute_segmentation_table,
    generate_binary_masks,
    merge_metadata_table,
    write_results_csv,
)
from mri_gist.fractal.labels import load_label_lookup
from mri_gist.registration.midway import (
    _apply_combined_transform,
    _compute_half_transform,
    hemisphere_split,
)


class PythonCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.data = np.zeros((16, 16, 16), dtype=np.int16)
        self.data[3:8, 4:12, 4:12] = 2
        self.data[8:13, 4:12, 4:12] = 41
        self.affine = np.diag([1.5, 2.0, 2.5, 1.0])
        self.input_path = self.directory / "sub-smoke_ses-01_seg.nii.gz"
        nib.save(nib.Nifti1Image(self.data, self.affine), self.input_path)

    def test_installed_entry_points_and_bundled_lut(self):
        executable = Path(sys.executable).with_name("mri-gist")
        for command in (
            [str(executable), "--help"],
            [sys.executable, "-m", "mri_gist", "--help"],
            [
                sys.executable,
                "-c",
                "from mri_gist.fractal.labels import load_label_lookup; "
                "assert load_label_lookup()[2] == 'Left-Cerebral-White-Matter'",
            ],
        ):
            with self.subTest(command=command):
                result = subprocess.run(
                    command,
                    cwd=self.directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if command[-1] == "--help":
                    self.assertIn("msp-align", result.stdout)
                    self.assertIn("fractal", result.stdout)

    def test_cli_command_help(self):
        runner = CliRunner()
        commands = [(name,) for name in cli.commands]
        commands.extend(("fractal", name) for name in ("compute", "batch", "masks"))
        for command in commands:
            with self.subTest(command=command):
                result = runner.invoke(cli, [*command, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Usage:", result.output)

    def test_runtime_module_imports(self):
        for module in (
            "nrrd",
            "psutil",
            "jinja2",
            "uvicorn",
            "mri_gist.config",
            "mri_gist.registration.core",
            "mri_gist.detection.hemisphere",
            "mri_gist.segmentation.synthseg",
            "mri_gist.backend.server",
            "mri_gist.visualization.server",
        ):
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_conversion_preserves_voxels_and_geometry(self):
        source = sitk.ReadImage(str(self.input_path))
        expected = sitk.DICOMOrient(source, "RAI")
        output_path = self.directory / "converted.nrrd"
        convert_format(str(self.input_path), str(output_path), "nrrd")
        converted = sitk.ReadImage(str(output_path))
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(converted), sitk.GetArrayFromImage(expected)
        )
        np.testing.assert_allclose(converted.GetSpacing(), expected.GetSpacing())
        np.testing.assert_allclose(converted.GetOrigin(), expected.GetOrigin())
        np.testing.assert_allclose(converted.GetDirection(), expected.GetDirection())

        batch_directory = self.directory / "batch"
        convert_format(str(self.directory), str(batch_directory), "nii.gz")
        batch_image = sitk.ReadImage(str(batch_directory / "converted.nii.gz"))
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(batch_image), sitk.GetArrayFromImage(expected)
        )

    def test_conversion_cleaning_and_missing_input(self):
        data = np.full((16, 16, 16), 10, dtype=np.int16)
        data[4:12, 4:12, 4:12] = 200
        input_path = self.directory / "noisy.nii.gz"
        nib.save(nib.Nifti1Image(data, self.affine), input_path)
        output_path = self.directory / "clean.nrrd"
        convert_format(str(input_path), str(output_path), "nrrd", clean_background=True)
        cleaned = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
        self.assertEqual(set(np.unique(cleaned)), {0, 200})
        self.assertEqual(np.count_nonzero(cleaned), 8 ** 3)
        with self.assertRaises(FileNotFoundError):
            convert_format(str(self.directory / "missing.nii"), str(output_path), "nrrd")

    def test_native_ants_registration(self):
        coordinates = np.indices((24, 24, 24), dtype=np.float32)
        distance = sum((coordinates[axis] - 11.5) ** 2 for axis in range(3))
        data = np.exp(-distance / 32).astype(np.float32)
        image = ants.from_numpy(data)
        result = ants.registration(
            fixed=image,
            moving=image.clone(),
            type_of_transform="Rigid",
            aff_iterations=(20, 10, 0, 0),
            random_seed=1,
            outprefix=str(self.directory / "registration_"),
        )
        warped = result["warpedmovout"].numpy()
        self.assertEqual(warped.shape, data.shape)
        self.assertTrue(np.isfinite(warped).all())
        self.assertGreater(np.count_nonzero(warped), 0)
        self.assertTrue(result["fwdtransforms"])
        for transform in result["fwdtransforms"]:
            self.assertTrue(Path(transform).is_file())
            self.assertTrue(np.isfinite(ants.read_transform(transform).parameters).all())

    def test_midway_math_and_hemisphere_geometry(self):
        transform = np.eye(4)
        transform[:3, 3] = [2, 4, 6]
        half = _compute_half_transform(transform)
        np.testing.assert_allclose(half @ half, transform, atol=1e-10)
        aligned = _apply_combined_transform(self.data, np.eye(4), order=0)
        np.testing.assert_array_equal(aligned, self.data)
        hemispheres = hemisphere_split(aligned, self.affine, lr_axis=0)
        np.testing.assert_array_equal(hemispheres["left_data"], self.data[:8])
        np.testing.assert_array_equal(hemispheres["right_data"], self.data[8:])
        np.testing.assert_allclose(
            hemispheres["right_img"].affine[:3, 3],
            self.affine[:3, 3] + self.affine[:3, 0] * 8,
        )
        self.assertEqual(
            hemispheres["left_nonzero"] + hemispheres["right_nonzero"],
            np.count_nonzero(self.data),
        )

    def test_fractal_results_masks_and_cli(self):
        lookup = load_label_lookup()
        table = compute_segmentation_table(self.input_path, per_label=True)
        self.assertEqual(list(table.columns), DEFAULT_RESULT_COLUMNS)
        self.assertEqual(list(table["scope"]), ["whole_brain", "label", "label"])
        self.assertEqual(list(table["nonzero_voxels"]), [640, 320, 320])
        self.assertEqual(list(table["label_name"]), ["whole_brain", lookup[2], lookup[41]])
        self.assertTrue(np.isfinite(table["fd"]).all())
        self.assertTrue(table["fd"].between(0, 3.5).all())
        self.assertEqual(set(table["participant_id"]), {"sub-smoke"})
        self.assertEqual(set(table["session_id"]), {"ses-01"})
        output_path = write_results_csv(table, self.directory / "results.csv")
        self.assertEqual(list(pd.read_csv(output_path).columns), DEFAULT_RESULT_COLUMNS)

        metadata_path = self.directory / "metadata.csv"
        metadata_path.write_text(
            "participant_id,session_id,group\nsub-smoke,ses-01,control\n",
            encoding="utf-8",
        )
        merged = merge_metadata_table(table, metadata_path)
        self.assertEqual(list(merged["group"]), ["control"] * 3)
        masks = generate_binary_masks(self.input_path, self.directory / "masks")
        self.assertEqual(len(masks), 2)
        for label, mask_path in zip((2, 41), masks):
            mask = nib.load(mask_path)
            np.testing.assert_array_equal(mask.get_fdata(), self.data == label)
            np.testing.assert_allclose(mask.affine, self.affine)

        cli_output = self.directory / "cli.csv"
        result = CliRunner().invoke(
            cli,
            ["fractal", "compute", str(self.input_path), "-o", str(cli_output), "--per-label"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        pd.testing.assert_frame_equal(pd.read_csv(cli_output), pd.read_csv(output_path))

    def test_analytics_and_api_schemas(self):
        from mri_gist.backend.analytics import MRIAnalytics
        from mri_gist.backend import server as backend
        from mri_gist.visualization import server as visualization

        statistics = MRIAnalytics(str(self.input_path)).basic_statistics()
        self.assertEqual(statistics["data_shape"], list(self.data.shape))
        self.assertAlmostEqual(statistics["basic_stats"]["mean"], float(self.data.mean()))
        self.assertAlmostEqual(statistics["volume_stats"]["voxel_volume_mm3"], 7.5)
        json.dumps(statistics, allow_nan=False)
        for server, expected_paths in (
            (backend, ("/api/health", "/api/process", "/api/analytics")),
            (visualization, ("/api/files", "/api/process/segment")),
        ):
            with self.subTest(server=server.__name__):
                schema = server.app.openapi()
                for path in expected_paths:
                    self.assertIn(path, schema["paths"])
                json.dumps(schema, allow_nan=False)
        with patch.object(
            backend,
            "JOB_REGISTRY",
            {
                "smoke-job": {
                    "status": "completed",
                    "task_type": "conversion",
                    "input_file": str(self.input_path),
                    "output_file": str(self.directory / "converted.nrrd"),
                    "timestamp": "2026-01-01T00:00:00",
                }
            },
        ):
            health = asyncio.run(backend.health_check())
            self.assertEqual(health["status"], "healthy")
            self.assertEqual(health["active_jobs"], 0)
            self.assertEqual(health["completed_jobs"], 1)
            response = asyncio.run(backend.get_processing_status("smoke-job"))
            payload = json.loads(response.model_dump_json())
            self.assertEqual(payload["job_id"], "smoke-job")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["task_type"], "conversion")
        files = asyncio.run(visualization.list_files(str(self.directory)))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, self.input_path.name)
        self.assertEqual(json.loads(files[0].model_dump_json())["type"], ".nii.gz")


if __name__ == "__main__":
    unittest.main()
