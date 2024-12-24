import click
import os

from lib.initialize_project import Project
from lib.logger import configure_logger


@click.command()
@click.option('-proj', '--project_path', help='Path to project folder.')
def run(project_path):
    os.chdir(project_path)
    logger = configure_logger(__name__, filename="LOG_Init.log", clear=True)

    # init the project
    config = Project(project_path, logger)
    logger.info(f"Project: {config.project}")

if __name__ == '__main__':
    run()