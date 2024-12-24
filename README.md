# rs-tiler
Generates tiles from spatial raster images and corresponding vector annotations. 

The entire process is managed by a configuration file. This is a YAML file.

To create tiles for your own project, you:

- Copy the projects/project_example dir 
- Add your tifs, annotations (polygons), optional BG (point)
- Make sure everything is in src 28992
- Update the config file
- Update the docker-compose.yml to point to your project
- Make sure the docker demon is running and then run the following command:

```bash
docker-compose up
```

- The tiles will be generated in the output dir. If you set```  save_tiles: true``` then the actual tiles are saved for visual inspection. 
- The final output is saved to the tiles folder as .npy train/val/test files.
- These can be used in your own tensorflow model (which does not to use the same normaliser as the one used in the tiler).
   