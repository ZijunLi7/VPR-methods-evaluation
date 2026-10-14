import os
import re
from collections import Counter
from glob import glob

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        if not images_paths:
            raise ValueError(f"Image path list is empty: {file_with_paths}; wait for preprocessing to complete")
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        print(f"Searching test images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [p for p in images_paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any images")
    return images_paths


def read_utms(paths):
    return np.asarray([(os.path.basename(path).split('@')[1], os.path.basename(path).split('@')[2])
                       for path in paths], dtype=float)


def split_paths_by_year(paths, query_years):
    """Read YYYYMM capture labels from the standard filename's timestamp field."""
    database, queries, counts, ids, zones = [], [], Counter(), set(), set()
    for path in paths:
        fields = os.path.basename(path).split('@')
        if len(fields) < 15 or not re.fullmatch(r'[0-9]{4}(?:0[1-9]|1[0-2])', fields[13]):
            raise ValueError(f"Expected YYYYMM timestamp in filename field 13: {path}")
        year = int(fields[13][:4])
        if year == 0 or not fields[7] or fields[7] in ids:
            raise ValueError(f"Invalid capture year or duplicate/empty panorama ID: {path}")
        ids.add(fields[7])
        zones.add((int(fields[3]), fields[4] >= 'N'))
        counts[year] += 1
        (queries if year in query_years else database).append(path)
    missing = set(query_years) - set(counts)
    if missing:
        raise ValueError(f"Query years absent from images: {sorted(missing)}")
    if len(zones) != 1:
        raise ValueError('Runtime year splitting requires one UTM zone and hemisphere')
    if not database or not queries:
        raise ValueError('Year split must contain both database and query images')
    return database, queries, dict(sorted(counts.items()))


def evenly_subsample_paths(paths, max_samples):
    """Return up to max_samples paths distributed over the full sequence."""
    if max_samples is None or max_samples >= len(paths):
        return paths
    indices = np.arange(max_samples, dtype=np.int64) * len(paths) // max_samples
    return [paths[index] for index in indices]


def subsample_database_paths(database_paths, query_paths, max_samples, use_labels):
    """Sample the database while retaining nearest entries for sampled queries."""
    if max_samples is None or max_samples >= len(database_paths):
        return database_paths
    if not use_labels:
        return evenly_subsample_paths(database_paths, max_samples)

    database_utms = read_utms(database_paths)
    query_utms = read_utms(query_paths)
    nearest = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(database_utms)
    nearest_indices = nearest.kneighbors(query_utms, return_distance=False).ravel()

    selected = list(dict.fromkeys(int(index) for index in nearest_indices))
    if len(selected) > max_samples:
        selected = evenly_subsample_paths(selected, max_samples)
    else:
        selected_set = set(selected)
        fill_candidates = np.arange(max_samples, dtype=np.int64) * len(database_paths) // max_samples
        for index in fill_candidates:
            index = int(index)
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                if len(selected) == max_samples:
                    break
        if len(selected) < max_samples:
            for index in range(len(database_paths)):
                if index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
                    if len(selected) == max_samples:
                        break

    return [database_paths[index] for index in sorted(selected)]


class TestDataset(data.Dataset):
    def __init__(
        self,
        database_folder=None,
        queries_folder=None,
        positive_dist_threshold=25,
        image_size=None,
        use_labels=True,
        max_database_samples=None,
        max_query_samples=None,
        images_folder=None,
        query_years=None,
        filter_dist_threshold=None,
    ):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()

        self.year_split_summary = None
        self.filtered_queries = []
        if images_folder is not None:
            if not query_years or not use_labels or database_folder is not None or queries_folder is not None:
                raise ValueError('Year splitting requires labelled images, query years, and no explicit split folders')
            all_database_paths, all_query_paths, year_counts = split_paths_by_year(read_images_paths(images_folder), query_years)
            self.year_split_summary = {'query_years': sorted(set(query_years)), 'year_counts': year_counts,
                                       'database_before_sampling': len(all_database_paths),
                                       'queries_before_sampling': len(all_query_paths)}
        else:
            if query_years is not None:
                raise ValueError('query_years requires images_folder')
            all_database_paths = read_images_paths(database_folder)
            all_query_paths = read_images_paths(queries_folder)
        self.queries_paths = evenly_subsample_paths(
            all_query_paths, max_query_samples
        )
        self.database_paths = subsample_database_paths(
            all_database_paths, self.queries_paths, max_database_samples, use_labels
        )

        if use_labels:
            # Read UTM coordinates, which must be contained within the paths
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            try:
                # This is just a sanity check
                image_path = self.database_paths[0]
                utm_east, utm_north = read_utms([image_path])[0]
            except:
                raise ValueError(
                    "The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
                    f"but it is {image_path}, which does not contain the UTM coordinates."
                )

            self.database_utms = read_utms(self.database_paths)
            self.queries_utms = read_utms(self.queries_paths)

            # Find positives_per_query, which are within positive_dist_threshold (default 25 meters).
            # Keep the fitted index so positives at additional thresholds can be
            # obtained without rebuilding it or extracting descriptors again.
            self._positives_knn = NearestNeighbors(n_jobs=-1)
            self._positives_knn.fit(self.database_utms)
            self._positives_cache = {}
            self.positive_dist_threshold = positive_dist_threshold
            if images_folder is not None:
                radius = positive_dist_threshold if filter_dist_threshold is None else filter_dist_threshold
                if not 0 < radius <= positive_dist_threshold:
                    raise ValueError('Filter distance must be positive and no larger than the evaluation radius')
                mask = np.array([len(p) > 0 for p in self.get_positives(radius)])
                rejected = np.flatnonzero(~mask)
                if len(rejected):
                    distances = self._positives_knn.kneighbors(self.queries_utms[rejected], n_neighbors=1)[0][:, 0]
                    self.filtered_queries = [{'image_path': self.queries_paths[i], 'nearest_database_distance_m': float(distance),
                                              'filter_threshold_m': radius, 'reason': 'no_positive_within_threshold'}
                                             for i, distance in zip(rejected, distances)]
                self.year_split_summary.update(database_after_sampling=len(self.database_paths),
                                               queries_before_filter=len(self.queries_paths),
                                               filter_threshold_m=radius, filtered_queries=len(rejected),
                                               retained_queries=int(mask.sum()))
                if not mask.any():
                    raise ValueError(f'All {len(mask)} queries have no database positives within {radius} m')
                self.queries_paths = [path for path, keep in zip(self.queries_paths, mask) if keep]
                self.queries_utms = self.queries_utms[mask]
                self._positives_cache = {}
            self.positives_per_query = self.get_positives(positive_dist_threshold)

        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.num_database = len(self.database_paths)
        self.num_queries = len(self.queries_paths)

        transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.transform = transforms.Compose(transformations)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.transform(pil_img)
        return normalized_img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< #queries: {self.num_queries}; #database: {self.num_database} >"

    def get_positives(self, positive_dist_threshold=None):
        """Return database positives for every query at the requested radius."""
        if positive_dist_threshold is None:
            positive_dist_threshold = self.positive_dist_threshold

        if positive_dist_threshold not in self._positives_cache:
            self._positives_cache[positive_dist_threshold] = self._positives_knn.radius_neighbors(
                self.queries_utms, radius=positive_dist_threshold, return_distance=False
            )
        return self._positives_cache[positive_dist_threshold]
