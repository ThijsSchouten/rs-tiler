import os
import yaml
import pprint

from lib.utils_os import create_folder
from lib.logger import configure_logger

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
        self.__dict__ = self

class Project:
    def __init__(self, project_folder, logger, config_fp="config.yml", validate=True, mkdirs=True):
        self.logger = logger
        self.logger.info(f"🍁 {project_folder}")

        self.project_folder = project_folder
        self.config_fp = config_fp
        self.config = self.load_config()

        self.logger.info(self)
        
        if mkdirs: self.mkdirs()
        if validate: self.validate()

    def load_config(self):
        # Assert path exists
        fp = self.config_fp
        assert os.path.exists(fp), "config.yml not found in project root."
        with open(fp, 'r') as f:
            return AttrDict(yaml.safe_load(f))
    
    def validate(self):
        assert len(self.files.images_train) > 0, "No images found in configfile."
        assert len(self.files.labels) > 0, "No labels found in configfile."
        assert isinstance(self.settings.tilesize, int) and self.settings.tilesize > 0, 'Invalid tilesize'
        assert isinstance(self.settings.split_ratio, list), "Invalid split_ratio"
        assert len(self.settings.split_ratio) == 3, "Supply train/val/test split ratios"
        assert round(sum(self.settings.split_ratio),3) == 1, f'Split ratio should sum to 1 -- {sum(self.settings.split_ratio)}'
        assert isinstance(self.settings.augment_ratio, (int, float)) and 0 <= self.settings.augment_ratio <= 1, 'Invalid augment_ratio'
        assert isinstance(self.settings.drop_bg_ratio, (int, float)) and 0 <= self.settings.drop_bg_ratio <= 1, 'Invalid drop_bg_ratio'
        assert isinstance(self.settings.strip_alpha_channel, bool), 'Invalid strip_alpha_channel'
        
        # Assert files exist
        for key in self.files.keys():
            files = self.files.get(key)
            if not files: continue
            files = [files] if isinstance(files, str) else files
            
            for p in files:
                if p == None: continue # Continue if null is specified
                assert os.path.exists(p), f"File {p} not found"

        self.logger.info("✅ Config Validation Passed.")

    def mkdirs(self):
        # if /cache path doesnt exist, create it
        # if /processing path doesnt exist, create it
        # if /results path doesnt exist, create it
        for key in self.dirs.keys():
            _dir = self.dirs.get(key)
            if create_folder(_dir):
                self.logger.info(f"Created dir {_dir}")


    def __getattr__(self, name):
        return self.config.get(name, None)

    def __repr__(self):
        return pprint.pformat(self.config)
    
# if __name__ == "__main__":
#     project_folder = "projects/proj-duizendknoop"  # Set this to your project folder path
#     os.chdir(project_folder)

#     logger = configure_logger("log.log", clear=True)
#     config = Project(project_folder, logger)