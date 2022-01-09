#### About
Toy monocular SLAM implementation for Creative Coding Practice M25723-2021/22

![image](https://user-images.githubusercontent.com/58654842/148688380-3e9b2c8a-82b4-4930-a7bb-7766195d965a.png)

&nbsp;

#### To Use:
```
git clone https://github.com/GandalfTea/slamhmw
cd slamhmw
pip install numpy PyOpenGL scikit-image opencv-contib-python pygame 
python slam.py [video].mp4
```
The video argument can be replaced with a number representing the index of a camera.  
Default index 0 will grab first camera found on system:
```
python slam.py 0
```
&nbsp;

Example video used:
```
https://www.youtube.com/watch?v=Auuf4lTvtSw
```

#### Todo:
* Write real-time version in C++.
* Camera argument to VideoCapture(sys.argv[1] does not work.
* Clean up points
* Use g2opy for cleanup?
