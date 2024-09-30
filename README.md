# Financial-CNN
This project takes 6 time series of financial data and passes them through a CNN for classification.

# Main Logic
Given a day in the financial market, we gather the last 120 close prices (scaled to 0-1) and find a historical period in time that best matches this vector.

![Matching](https://github.com/user-attachments/assets/c29d9c72-d592-4f4e-8726-095b95c00676)

Now that we have some dates $(x,y)$ which represent a period of similar price action, we format the period $[x+50, y+50]$ as our training set. Namely, we are intersted in the Open, High, Low, Close, Volume, and 14D momentum for each of the days in $[x+50, y+50]$.

We then slide a window of size 20 over this data to get training instances in $[0,1]^{20\times6}$. These matrices can be visualized and look as follows:

![20x6_matrix_visualized](https://github.com/user-attachments/assets/9586a9ae-a3ed-4c1e-91ad-e5576c2af99f)

We then either pass these images to a pre-existing CNN (that gets more training data sequentially), or fit a new CNN to this data.

After training, we pass in the image representing the past 20 day's worth of Open, High, Low, Close, Volume, and 14D momentum data for classification.

We can play with the threshold for classification (0.9 seems to work well, but misses a lot of winning trades).

# Instructions
pip install any packages that may be missing on your device and run `main.ipynb`. This build is stable with Python 3.9.18 and Cuda 12.6.
