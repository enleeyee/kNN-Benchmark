This project is based on vincentfpgarcia/kNN-CUDA. Extensions and benchmarking modifications made by Enlai Yii.

To run tests:
```
make clean && make test
./test
```

To run the WebGPU: 
```
python3 -m http.server 8000
```

Then open localhost via port 8000 as below:
```
http://localhost:8000/webgpu_knn.html 
```
NOTE: Recommend browsers that can run WebGPU like Google Chrome Canary. 
