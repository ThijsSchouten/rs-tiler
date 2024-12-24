# Standard Library Imports
import click
import os
import shutil
from glob import glob
import numpy as np

# Third Party Imports
import geopandas as gpd
from shapely.geometry import box
import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# Plotting
import matplotlib.pyplot as plt

# Local Imports
from lib.initialize_project import Project
from lib.logger import configure_logger
from lib.decorators import timer
from lib.utils import distribute_indices
from lib.utils import chw_to_hwc, remove_alpha_if_exists, check_nodata
from lib.augmenter import augment, save_images


class Tiler:
    def __init__(
        self,
        img_fp,
        bg_fp,
        aoi_fp,
        lbl_fp,
        out_path,
        tiles_path,
        tilesize,
        splitsize=[0.7, 0.2, 0.1],
        augment_ratio=0,
        drop_bg_ratio=0,
        save_tiles=False,
        logger=None,
    ):
        self.logger = logger

        self.logger.info("Initializing Tiler")
        self.logger.info(f"Tilesize {tilesize}, splitsize {splitsize}")

        self.out_path = out_path
        self.tiles_path = tiles_path
        self.save_tiles = save_tiles

        if save_tiles:
            self.logger.warning(
                "Saving tiles is enabled. This will create PNG for each tile. This will slow down the process."
            )
            fp = f"{self.tiles_path}/tiles"
            self.logger.info(f"Recreating tiles folder {fp}")
            if os.path.exists(fp):
                shutil.rmtree(fp)
            os.makedirs(fp, exist_ok=True)

        self.tilesize = tilesize
        self.tilestride = tilesize  # TODO add stride option.
        self.splitsize = splitsize
        self.augment_ratio = augment_ratio
        self.drop_bg_ratio = drop_bg_ratio

        self.tilewindows = []
        self.aoi_geom = False
        self.bg_geom = False

        self.crs = None

        # TODO support multiple images
        self.tifs = img_fp
        self.lbls = lbl_fp
        self.aoi = aoi_fp
        self.bgs = bg_fp

        self.tiledata = {
            "train": {"images": [], "masks": []},
            "val": {"images": [], "masks": []},
            "test": {"images": [], "masks": []},
        }

    def set_bg(self, bg_fp):
        """Loads the area of interest is specified. If a background is specified, it is merged with the AOI."""
        # check if bg_fp file exists:
        self.logger.info(f"Loading background from {bg_fp}")
        bg = gpd.read_file(bg_fp)
        self.bg_geom = bg.dissolve().geometry[0]

    def load_labels(self, fp):
        """Loads the labels from the specified files."""
        # Todo add support to load only 1 label with a
        self.labels = gpd.read_file(fp).dissolve()
        self.logger.info(f"Loaded labels from {fp}")

    def generate_tilewindows(self, tif):
        self.logger.info(f"Creating tilewindows from {tif}")
        # open the image with rio
        with rio.open(tif) as src:
            self.crs = src.crs
            # create tilewindows
            for row in range(0, src.height, self.tilestride):
                for col in range(0, src.width, self.tilestride):
                    window = rio.windows.Window(col, row, self.tilesize, self.tilesize)
                    bounds = src.window_bounds(window)
                    bbox = box(*bounds)

                    # Calculate the overlap percentage of the labels with the windows
                    intersection = self.labels[self.labels.intersects(bbox)]
                    overlap_area = sum(
                        label.intersection(bbox).area for label in intersection.geometry
                    )
                    overlap_prc = overlap_area / bbox.area

                    # Create dict holding the tile info
                    tiledict = {
                        "dataset": tif,
                        "overlap": overlap_prc,
                        "geom": bbox,
                        "split": "train",
                        "augment_count": 0,
                        "user_specified_bg": False,
                    }
                    tiledict.update(window.todict())  # add the window info

                    if self.bg_geom and bbox.intersects(self.bg_geom):
                        tiledict["user_specified_bg"] = True

                    self.tilewindows.append(tiledict)

        self.logger.info(f" Added {len(self.tilewindows)} tilewindows")

    def convert_tilewindows_to_gdf(self):
        # Convert the list of dicts to a GeoDataFrame
        self.logger.info("Converting tilewindows to a GeoDataFrame")
        self.tilewindows = gpd.GeoDataFrame(
            self.tilewindows, crs=self.crs, geometry="geom"
        )
        self.drop_empty_tiles()
        self.append_splits()
        self.append_augmentations()

    def save_tilewindow_gdf(self):
        fp = self.tiles_path + "/tilewindows.gpkg"
        self.tilewindows.to_file(fp, driver="GPKG")
        self.logger.info(f"🚀 Saved tilewindows to {fp}")

    def drop_empty_tiles(self):
        """Drops tiles without overlap with the labels. The chance of
        dropping a tile is determined by the self.drop_bg_ratio. Tiles with
        user specified background are never dropped.
        """
        self.logger.info("Dropping empty tiles")
        tw = self.tilewindows

        # Select the tiles without overlap and without user specified background
        bg_only = tw[(tw["overlap"] == 0) & (tw["user_specified_bg"] == False)]

        # Sample these tiles based on the drop_bg_ratio
        drop_idx = bg_only.sample(frac=self.drop_bg_ratio).index

        self.logger.info(f"Dropping {len(drop_idx)} / {len(self.tilewindows)} tiles")

        self.tilewindows.drop(drop_idx, inplace=True)
        self.tilewindows.reset_index(drop=True, inplace=True)

    def append_splits(self):
        """
        Appends the split to the record, based on the self.splitsize splitsizes. The splitsize ratios determines the percentage of the total overlap in each split.
        Tiles without overlap are evenly (randomly) distributed among the splits.
        """
        self.logger.info(f"Appending splits to tilewindows")

        values = self.tilewindows["overlap"].to_list()
        ratios = self.splitsize
        train_idx, val_idx, test_idx = distribute_indices(values, ratios)
        # update the split columns
        self.tilewindows.loc[train_idx, "split"] = "train"
        self.tilewindows.loc[val_idx, "split"] = "val"
        self.tilewindows.loc[test_idx, "split"] = "test"

    def append_augmentations(self):
        """
        Appends the number of augmentations to the record, based on the self.augment_ratio.
        """
        self.logger.info(f"Appending augmentations to tilewindows")
        # if augment ratio is 0, return
        if self.augment_ratio == 0:
            return

        # update aug attribute. Per record, the number of augmentations
        # is determined by random chance (self.augment_ratio). Augmentation
        # is only applied to the training set.
        self.tilewindows["augment_count"] = np.random.binomial(
            1, self.augment_ratio, len(self.tilewindows)
        )
        self.tilewindows.loc[self.tilewindows["split"] != "train", "augment_count"] = 0
        # self.tilewindows.loc[self.tilewindows['overlap'] == 0, 'augment_count'] = 0

    def load_tiledata(self, tif):
        """Loop through the self.tilewindows.
        Per tilewindow, extract the tile from the image.
        Create a mask with 0 as BG and 1 as label.
        """
        self.logger.info(f"Reading image data from {tif}")

        # select tilwindows where dataset == tif
        dataset_windows = self.tilewindows[self.tilewindows["dataset"] == tif]

        with rio.open(tif) as src:
            for idx, tile in dataset_windows.iterrows():
                # read the tile from the image
                window = rio.windows.Window.from_slices(
                    (tile["row_off"], tile["row_off"] + tile["height"]),
                    (tile["col_off"], tile["col_off"] + tile["width"]),
                )
                img = src.read(window=window, boundless=True, fill_value=0)

                img = chw_to_hwc(img)
                img = remove_alpha_if_exists(img)

                # Create an empty mask (duplicate img and set values to 0)
                # Burn-in the labels, 1 for label, 0 for BG
                msk = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                shapes = [(geom, 1) for geom in self.labels.geometry]
                msk = rasterize(
                    shapes=shapes,
                    out=msk,
                    transform=src.window_transform(window),
                    dtype=np.uint8,
                    all_touched=True,
                )

                # Append the image and mask to the correct list
                # img_norm = normalize(img)
                split = tile["split"]
                self.tiledata[split]["images"].append(img)
                self.tiledata[split]["masks"].append(msk)

                # If augment_count > 0, append the augmented tiles
                if tile["augment_count"] == 0:
                    if self.save_tiles:
                        save_images(f"{self.tiles_path}/tiles/{idx}_{split}", img, msk)
                    continue

                aug_img, aug_msk = augment(img, msk)
                if self.save_tiles:
                    save_images(
                        f"{self.tiles_path}/tiles/{idx}_aug_{split}",
                        img,
                        msk,
                        aug_img,
                        aug_msk,
                    )

                # aug_img_norm = normalize(aug_img)
                self.tiledata[tile["split"]]["images"].append(aug_img)
                self.tiledata[tile["split"]]["masks"].append(aug_msk)

    def save_npy_out(self):
        # Save the tiles and masks for each split
        for split_name, content in self.tiledata.items():
            if len(content["images"]) > 0:
                tiles = np.stack(content["images"])
                masks = np.expand_dims(np.stack(content["masks"]), axis=-1)

                # Generate a random permutation of indices
                indices = np.random.permutation(tiles.shape[0])

                # Shuffle the tiles and masks using the same indices
                tiles = tiles[indices]
                masks = masks[indices]

                # Plot the datatype
                np.save(f"{self.tiles_path}/{split_name}_images.npy", tiles)
                np.save(f"{self.tiles_path}/{split_name}_masks.npy", masks)
                self.logger.info(
                    f"Saved {tiles.shape} {split_name} images to {self.out_path}"
                )
                self.logger.info(
                    f"Saved {masks.shape} {split_name} masks to {self.out_path}"
                )

    def run(self):
        # Per image, label, bg combination, generate the tilewindows
        for tif_fp, lbl_fp, bg_fp in zip(self.tifs, self.lbls, self.bgs):
            self.bg_geom = False
            if bg_fp != None:
                self.set_bg(bg_fp)
            self.load_labels(lbl_fp)
            self.generate_tilewindows(tif_fp)

        self.convert_tilewindows_to_gdf()
        self.save_tilewindow_gdf()  # Saves the GDF to disk

        for tif_fp, lbl_fp in zip(self.tifs, self.lbls):
            self.load_labels(lbl_fp)
            self.load_tiledata(tif_fp)  # Generates the tiles

        self.save_npy_out()


@click.command()
@click.option("-proj", "--project_path", help="Path to project folder.")
def run(project_path):
    os.chdir(project_path)
    logger = configure_logger(__name__, filename="LOG_Tiler.log", clear=True)

    # init the project
    config = Project(project_path, logger)
    logger.info(f"Project: {config.project}")

    tiler = Tiler(
        img_fp=config.files.images_train,
        lbl_fp=config.files.labels,
        aoi_fp=config.files.aoi,
        bg_fp=config.files.bg,
        out_path=config.dirs.results,
        tiles_path=config.dirs.tiles,
        tilesize=config.settings.tilesize,
        splitsize=config.settings.split_ratio,
        augment_ratio=config.settings.augment_ratio,
        drop_bg_ratio=config.settings.drop_bg_ratio,
        save_tiles=config.settings.save_tiles,
        logger=logger,
    )

    tiler.run()


if __name__ == "__main__":
    run()
