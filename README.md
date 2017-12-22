# K-means compression algorithm
A python implementation of the k-means clustering algorithm to compress png and jpg files.

![picture](sample.jpg)
Before - 507 KB

![picture](test.jpg)
After (with 32 colours) - 145 KB

# Prerequisites
- Python3.5 or above
- To install, run
```bash
sudo apt-get install python3
```
## Python3 packages
- numpy
- matplotlib
- PIL
- scipy
- To install all these, run
```bash
sudo -H pip3 install numpy matplotlib PIL scipy
```

# How to run
```bash
python3 kmeans.py <image-name>
```
The number of colors can be modified in the kmeans.py file line #91

