import os 

import click
# import rasterio as rio
import tensorflow as tf

from lib.initialize_project import Project
from lib.logger import configure_logger
from lib.decorators import timer

class RasterPredictor():
    def __init__(self, tifs, model_fp, threshold, tilesize,logger):
        self.logger = logger
        self.tifs = tifs
        self.model_fp = model_fp
        self.threshold = threshold
        self.tilesize = tilesize

    def load_model(self):
        self.logger.info(f"Loading model {self.model_fp}")
        self.model = tf.keras.models.load_model(self.model_fp)
        self.logger.info(f"Model loaded")

    def predict(self):
        self.load_model()
        for tif in self.tifs:
            self.logger.info(f"Predicting {tif}")
            


@click.command()
@click.option('-proj', '--project_path', help='Path to project folder.')
def run(project_path):
    os.chdir(project_path)
    logger = configure_logger(__name__, filename="LOG_Predictor.log", clear=True)

    # init the project
    config = Project(project_path, logger)
    logger.info(f"Project: {config.project_path}")

    predictor = RasterPredictor(
        tifs=config.files.images_predict,
        model_fp='models/cp-0022.ckpt',
        threshold=config.settings.prediction_threshold,
        tilesize=config.settings.tilesize,
        logger=logger
    )

    predictor.predict()



if __name__ == '__main__':
    run()