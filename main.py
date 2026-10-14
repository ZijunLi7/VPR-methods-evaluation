import parser
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset


def collate_images_by_shape(batch):
    """Stack images of equal shape while preserving a larger logical batch.

    Some datasets contain multiple native camera resolutions. A standard
    DataLoader collate fails when a batch crosses a resolution boundary, while
    resizing would alter the evaluation images. Returning one tensor per shape
    keeps native resolution and only splits the exceptional mixed batch.
    """
    groups = {}
    for image, index in batch:
        images, indices = groups.setdefault(tuple(image.shape), ([], []))
        images.append(image)
        indices.append(index)
    return [(torch.stack(images), torch.as_tensor(indices)) for images, indices in groups.values()]


def main(args):
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        # Use the largest threshold as the default for saved visualizations.
        positive_dist_threshold=max(args.positive_dist_threshold),
        image_size=args.image_size,
        use_labels=args.use_labels,
        max_database_samples=args.max_database_samples,
        max_query_samples=args.max_query_samples,
        images_folder=args.images_folder,
        query_years=args.query_years,
        filter_dist_threshold=min(args.positive_dist_threshold),
    )
    if test_ds.year_split_summary is not None:
        logger.info(f'Year split and closed-set query filtering: {test_ds.year_split_summary}')
        (log_dir / 'year_split_summary.json').write_text(json.dumps(test_ds.year_split_summary, indent=2) + '\n')
        with (log_dir / 'filtered_queries.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_path', 'nearest_database_distance_m', 'filter_threshold_m', 'reason'])
            writer.writeheader()
            writer.writerows(test_ds.filtered_queries)
        for split, paths in [('database', test_ds.database_paths), ('queries', test_ds.queries_paths)]:
            (log_dir / f'{split}_images_paths.txt').write_text('\n'.join(paths) + '\n')

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)
    if args.max_database_samples is not None or args.max_query_samples is not None:
        logger.warning("Sample limits are enabled; Recall values are for a smoke-test subset, not the full dataset")
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            collate_fn=collate_images_by_shape,
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for image_batches in tqdm(database_dataloader):
            for images, indices in image_batches:
                descriptors = model(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            collate_fn=collate_images_by_shape,
        )
        for image_batches in tqdm(queries_dataloader):
            for images, indices in image_batches:
                descriptors = model(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    # For each query, check if the predictions are correct
    if args.use_labels:
        for positive_dist_threshold in args.positive_dist_threshold:
            positives_per_query = test_ds.get_positives(positive_dist_threshold)
            recalls = np.zeros(len(args.recall_values))
            for query_index, preds in enumerate(predictions):
                for i, n in enumerate(args.recall_values):
                    if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                        recalls[i:] += 1
                        break

            # Divide by num_queries and multiply by 100, so the recalls are in percentages
            recalls = recalls / test_ds.num_queries * 100
            recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
            logger.info(f"Positive distance threshold {positive_dist_threshold} m: {recalls_str}")

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
