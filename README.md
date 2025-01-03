# VO Pipeline by Cedic Keller and Yves Inglin

## Running the Pipeline

1. **Prepare the Dataset**  
   Download the datasets and extract them into the `data` folder:
   - [PARKING](https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip)
   - [KITTI](https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip)
   - [MALAGA](https://rpg.ifi.uzh.ch/docs/teaching/2024/malaga-urban-dataset-extract-07.zip)

2. Set up the environment:
   - Install [Anaconda](https://www.anaconda.com/products/distribution).
   - Create the environment:
     ```bash
     conda env create -f environment.yml
     ```
   - Activate the environment:
     ```bash
     conda activate vo-env
     ```
3. Run the pipeline:
   ```bash
   python main.py
   ```
4. Select the dataset in `main.py` by setting the `DATASET` variable:
   ```python
   DATASET = Dataset.PARKING  # or Dataset.KITTI, Dataset.MALAGA
   ```

## Library Methods

- **Bootstrapping:**
  - `cv2.goodFeaturesToTrack`: Shi-Tomasi corner detection
  - `cv2.calcOpticalFlowPyrLK`: Lucas-Kanade tracking
  - `cv2.findFundamentalMat`: 8-point (normalized) RANSAC for structure from motion (SfM)
  - Alternative (not longer used): SIFT features and descriptors for correspondences

- **Continuous Operation:**
  - `cv2.calcOpticalFlowPyrLK`: Keypoint and candidate tracking
  - `cv2.solvePnPRansac`: P3P RANSAC for 2D-3D Camera localization
  - `cv2.Rodrigues`: Rotation vector to matrix conversion
  - `cv2.triangulatePoints`: Landmark triangulation
  - `cv2.goodFeaturesToTrack`: Shi-Tomasi for detction of new candidate points

- **Referenced Code:**  
  The `previous` folder contains code from earlier exercises for triangulation and pose disambiguation during bootstrapping.

## Screencasts

The pipeline runs on a single thread and was tested on a Dell XPS-15 laptop with an Intel i7-12700H @ 4.7 GHz with 32GB RAM.

- [PARKING](https://youtu.be/rg94vY-mSGI)
- [KITTI](https://youtu.be/clEpib1DddE)
- [MALAGA](https://youtu.be/nEUaAfyF-UQ)