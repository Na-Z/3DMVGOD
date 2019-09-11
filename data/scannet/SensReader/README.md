# Data Exporter

Usage:

For reading one scan, run
```
python2 reader.py --filename [.sens file to export data from] --output_path [output directory to export data to] --frame_skip [num frames to skip]
Options:
--export_depth_images: export all depth frames as 16-bit pngs (depth shift 1000)
--export_color_images: export all color frames as 8-bit rgb jpgs
--export_poses: export all camera poses (4x4 matrix, camera to world)
--export_intrinsics: export camera intrinsics (4x4 matrix)
```

If reading all the scans, run
```
python2 bash_reader.py --src_dir [input directory storing .sens files] --des_dir [output directory to export data to] --scans_list_file [the file path of scan names list]
```


The ScanNet dataset is organized as follows:

        Datasets/Scannet/scans/

            scene****_00/
                color/ 
                    |-- <frameID>.jpg
                        RGB frames export from .sens
                pose/
                    |-- <frameID>.txt 
                        Camera pose parameters export from .sens
                intrinsic/
                    |--intrinsic_color.txt
                       Color Camera intrinsic parameters before calibration (e.g., undistoration)
                    |--intrinsic_depth.txt
                    |--extrinsic_color.txt
                    |--extrinsic_deoth.txt
                instance-filt/ 
                    |-- <frameID>.png 
                        2D projections of aggregated annotation instances as 8-bit pngs
                |-- <scanId>.sens
                    RGB-D sensor stream containing color frames, depth frames, camera poses and other data
                |-- <scanId>_vh_clean.ply
                    High quality reconstructed mesh
                |-- <scanId>_vh_clean_2.ply
                    Cleaned and decimated mesh for semantic annotations
                |-- <scanId>_vh_clean_2.0.010000.segs.json
                    Over-segmentation of annotation mesh
                |-- <scanId>.aggregation.json, <scanId>_vh_clean.aggregation.json
                    Aggregated instance-level semantic annotations on lo-res, hi-res meshes, respectively
                |-- <scanId>_vh_clean_2.0.010000.segs.json, <scanId>_vh_clean.segs.json
                    Over-segmentation of lo-res, hi-res meshes, respectively (referenced by aggregated semantic annotations)
                |-- <scanId>_vh_clean_2.labels.ply
                    Visualization of aggregated semantic segmentation; colored by nyu40 labels (see img/legend; ply property 'label' denotes the ScanNet label id)
                |-- <scanId>_2d-label.zip
                    Raw 2d projections of aggregated annotation labels as 16-bit pngs with ScanNet label ids
                |-- <scanId>_2d-instance.zip
                    Raw 2d projections of aggregated annotation instances as 8-bit pngs
                |-- <scanId>_2d-label-filt.zip
                    Filtered 2d projections of aggregated annotation labels as 16-bit pngs with ScanNet label ids
                |-- <scanId>_2d-instance-filt.zip
                    Filtered 2d projections of aggregated annotation instances as 8-bit pngs