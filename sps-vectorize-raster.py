import os
import click
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import shape
from glob import glob
import time

from lib.initialize_project import Project
from lib.logger import configure_logger
from lib.decorators import timer


class Vectorizer:
    def __init__(
        self, logger, tifpath, target, threshold, simplify=False, filetypes="gpkg"
    ):
        self.logger = logger
        self.target = target
        self.tifs = self.get_tif_filepaths(tifpath)
        self.threshold = threshold
        self.crs = None
        self.filetypes = filetypes
        self.simplify = simplify

    def get_tif_filepaths(self, path):
        """Get all tif filepaths in a directory."""
        # self.logger.info(os.listdir(self))
        self.logger.info("Looking for tif files in: " + path)

        tifs = glob(os.path.join(path, "*.tif"))[::-1]
        # tifs_subfolders = glob(os.path.join(path, "*", "*.tif"))

        # tifs = tifs + tifs_subfolders

        # Get the files in the target folder
        files_in_target_path = glob(os.path.join(self.target, "*"))[::-1]
        target_path_files_basenames = [
            os.path.basename(t).split(".")[0] for t in files_in_target_path
        ]

        skips = [
            t
            for t in tifs
            if os.path.basename(t).split(".")[0] in target_path_files_basenames
        ]
        tifs = [
            t
            for t in tifs
            if os.path.basename(t).split(".")[0] not in target_path_files_basenames
        ]

        # check if basename is not part of any filename in target folder
        self.logger.info(f"Found {len(tifs)} tif files in {path}")
        self.logger.info(f"Skipped {len(skips)} tif files in {path}")
        self.logger.info(f"Skipping files: {skips}")
        return tifs

    def run(self, image_ids=[]):
        if image_ids == []:
            images = self.tifs
        else:
            images = [t for i, t in enumerate(self.tifs) if i in image_ids]

        self.logger.info(f"Starting vectorization of {len(images)} image(s)")

        for img in images:
            self.vectorize(img, self.threshold)

    def vectorize(self, img, threshold, tile_size=2048):
        self.logger.info(f"Vectorizing {img}, [threshold: {threshold}]")
        start = time.time()

        results = []

        with rio.open(img) as src:
            crs = src.crs
            width = src.width
            height = src.height

            # Get windows
            windows = []
            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    window = rio.windows.Window(
                        j, i, min(tile_size, width - j), min(tile_size, height - i)
                    )
                    windows.append(window)

            # Loop through windows
            for window in windows:
                # Read the image in tiles instead of the whole image
                image = src.read(1, window=window)

                # Create mask based on threshold
                mask = np.where(image > threshold, 1, 0)
                if mask.sum() == 0:
                    continue
                mask = mask.astype(np.int16)

                # Generate shapes only for the cells that meet condition
                results += [
                    {"geometry": shape(geom), "class": value}
                    for geom, value in shapes(
                        mask, transform=src.window_transform(window)
                    )
                    if value == 1
                ]

                self.logger.info(f"{window} Processed {len(results)} shapes")

        # Results is a list of geometries. Convert to geodataframe
        geometries = [item["geometry"] for item in results]

        if len(geometries) == 0:
            self.logger.info("No shapes found")
            return
        gdf = gpd.GeoDataFrame(geometry=geometries)
        gdf["class"] = 1

        # Dissolve the shapes that touch each other
        self.logger.info("Dissolving shapes")
        gdf = gdf.dissolve(by="class")
        self.logger.info("Skipping simplify")
        # gdf = gdf.simplify(tolerance=0.02, preserve_topology=True)
        gdf = gdf.explode()

        # Convert GDF from multiple polygons to single polygon
        self.logger.info("Converting to single polygon")

        self.logger.info(gdf)

        target_fp = os.path.join(self.target, os.path.basename(img))
        self.save_vector(gdf, crs, target_fp)

        duration = time.time() - start
        self.logger.info(f"Vectorization of {img} took {duration:.2f} seconds")

    def save_vector(self, gdf, crs, target_fp):
        # If gpkg in self.filetypes, save as gpkg
        if "gpkg" in self.filetypes:
            gpkg_fp = target_fp.replace(".tif", ".gpkg")
            gdf.to_file(gpkg_fp, crs=crs, driver="GPKG")
            self.logger.info(f"\n -> WRITTEN AS GPKG TO: {gpkg_fp}")
        if "geojson" in self.filetypes:
            geojson_fp = target_fp.replace(".tif", ".geojson")
            gdf.to_file(geojson_fp, crs=crs, driver="GeoJSON")
            self.logger.info(f"\n -> WRITTEN AS GEOJSON TO: {geojson_fp}")


@click.command()
@click.option("-proj", "--project_path", help="Path to project folder.")
def run(project_path):
    os.chdir(project_path)
    logger = configure_logger(__name__, filename="LOG_Vectorizer.log", clear=True)

    # init the project
    config = Project(project_path, logger)
    # logger.info(f"Project: {config.project}")

    vectorizer = Vectorizer(
        logger,
        tifpath=config.dirs.results,
        target=config.dirs.vectors,
        threshold=config.settings.prediction_threshold,
        filetypes=["gpkg"],  # "geojson",
    )

    vectorizer.run()


if __name__ == "__main__":
    run()
