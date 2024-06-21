# onnx转换和测试说明
## onnx转换
```Shell
python convert_onnx.py
```
## onnx测试

- 单张图像测试
  ```Shell
    python test_onnx.py --image ../dataset/02_6178896901985513.jpg
    ```
- 多张图像测试
  ```Shell
    python test_onnx.py  --image  /media/xin/data/data/seg_data/ours/ORIGIN/20240617_wire/test_select.txt
    ``` 
