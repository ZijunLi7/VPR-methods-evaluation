"""Runtime capture-year splitting, closure after sampling, and legacy behavior."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from test_dataset import TestDataset
import parser


def image_name(pid, east, year):
    return f'@{east:.3f}@0.000@51@R@25.0@121.55@{pid}@00@@@@@{year}02@yaw_rel_0_pitch_rel_0@.jpg'


class YearSplitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.all = self.root / 'images/all'
        self.all.mkdir(parents=True)
        self.samples = [('d0', 0, 2017), ('d1', 100, 2018), ('q0', 5, 2024),
                        ('q1', 120, 2025), ('qbad', 300, 2025)]
        for pid, east, year in self.samples:
            Image.new('RGB', (32, 32), 'green').save(self.all / image_name(pid, east, year))

    def dataset(self, **options):
        return TestDataset(images_folder=str(self.all), query_years=[2024, 2025], **options)

    def test_year_split_and_closed_set(self):
        ds = self.dataset(positive_dist_threshold=25)
        self.assertEqual((ds.num_database, ds.num_queries), (2, 2))
        self.assertEqual(len(ds.filtered_queries), 1)
        self.assertIn('@qbad@', ds.filtered_queries[0]['image_path'])
        self.assertEqual(ds.filtered_queries[0]['nearest_database_distance_m'], 200)
        self.assertTrue(all(len(p) for p in ds.positives_per_query))
        self.assertEqual(ds[ds.num_database][1], 2)
        self.assertEqual(tuple(ds[0][0].shape), (3, 32, 32))

    def test_smallest_radius_and_filter_after_database_sampling(self):
        ds = self.dataset(positive_dist_threshold=25, filter_dist_threshold=10)
        self.assertEqual(ds.num_queries, 1)
        self.assertEqual(len(ds.filtered_queries), 2)
        self.assertTrue(all(len(p) for p in ds.get_positives(10)))
        self.assertTrue(all(len(p) for p in ds.get_positives(25)))
        ds = self.dataset(positive_dist_threshold=25, max_database_samples=1)
        self.assertEqual((ds.num_database, ds.num_queries), (1, 1))
        self.assertTrue(all(len(p) for p in ds.positives_per_query))

    def test_legacy_folders_do_not_apply_year_filter(self):
        db, queries = self.root / 'db', self.root / 'queries'
        db.mkdir(); queries.mkdir()
        Image.new('RGB', (32, 32)).save(db / image_name('d0', 0, 2017))
        Image.new('RGB', (32, 32)).save(queries / image_name('qbad', 300, 2025))
        ds = TestDataset(str(db), str(queries))
        self.assertEqual(ds.num_queries, 1)
        self.assertIsNone(ds.year_split_summary)
        self.assertEqual(len(ds.positives_per_query[0]), 0)

    def test_empty_invalid_and_duplicate_splits(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            TestDataset(images_folder=str(self.all), query_years=[2023])
        with self.assertRaisesRegex(ValueError, 'both database'):
            TestDataset(images_folder=str(self.all), query_years=[2017, 2018, 2024, 2025])
        with self.assertRaisesRegex(ValueError, 'All .* queries'):
            self.dataset(positive_dist_threshold=1)
        Image.new('RGB', (32, 32)).save(self.all / image_name('q0', 10, 2019))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.dataset()

    def test_cli_modes(self):
        with patch.object(sys, 'argv', ['main.py', '--images_folder', str(self.all), '--query_years', '2025', '2024']):
            args = parser.parse_arguments()
        self.assertEqual(args.query_years, [2024, 2025])
        for extra in [['--no_labels'], ['--database_folder', 'db'], ['--queries_folder', 'q']]:
            with patch.object(sys, 'argv', ['main.py', '--images_folder', str(self.all), '--query_years', '2025', *extra]):
                with self.assertRaises(SystemExit):
                    parser.parse_arguments()

    def test_main_pipeline_records_filtered_queries(self):
        import main
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2, 3))[:, None].repeat(1, 512)
        arguments = ['main.py', '--images_folder', str(self.all), '--query_years', '2024', '2025',
                     '--method', 'cosplace', '--backbone', 'ResNet18', '--descriptors_dimension', '512',
                     '--device', 'cpu', '--num_workers', '0', '--batch_size', '2',
                     '--positive_dist_threshold', '10', '25', '--recall_values', '1', '2',
                     '--log_dir', str(self.root / 'logs')]
        with patch.object(sys, 'argv', arguments):
            args = parser.parse_arguments()
        with patch.object(main.vpr_models, 'get_model', return_value=DummyModel()):
            main.main(args)
        summary_path = next((self.root / 'logs').rglob('year_split_summary.json'))
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary['filter_threshold_m'], 10)
        self.assertEqual(summary['retained_queries'], 1)
        self.assertEqual(summary['filtered_queries'], 2)
        self.assertTrue((summary_path.parent / 'filtered_queries.csv').exists())


if __name__ == '__main__':
    unittest.main()
