### Train IDs

Regardless of the source data, the softmax output of the segmentation network follows Cityscapes class indexing:

| ID | Class |
|---|---|
| 0 | road |
| 1 | sidewalk |
| 2 | building |
| 3 | wall |
| 4 | fence |
| 5 | pole |
| 6 | traffic light |
| 7 | traffic sign |
| 8 | vegetation |
| 9 | terrain |
| 10 | sky |
| 11 | person |
| 12 | rider |
| 13 | car |
| 14 | truck |
| 15 | bus |
| 16 | train |
| 17 | motorcycle |
| 18 | bicycle |

Since the stored IDs in the segmentation masks in SYNTHIA, GTA5 and Cityscapes are inconsistent, one needs to convert them to the same indexing.

### Converting ground truth to train IDs
It is possible to re-map the class indices of the segmentation masks directly in the dataloader and to load the original ground-truth maps.
We pre-computed this mapping offline, however.
The script `tools/convert_train_ids.py` reads in the original ground-truth masks, remaps the class IDs and saves the result on disk.
To run the script, you can use the following template:
```bash
python tools/convert_train_ids.py --dataset [cs|gta|synthia]
                                  --ann-data [path/to/labels/]
                                  --ann-out [output/directory/] 
```
