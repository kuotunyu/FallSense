# Third-party notices

FallSense 本身以 MIT License 發行。安裝與 container build 會取得第三方 Python
packages；各 package 仍適用其自身授權條款。主要 runtime/training 依賴如下：

| component | license | project |
|---|---|---|
| NumPy | BSD-3-Clause（binary wheel 另含其 bundled notices） | https://numpy.org |
| pandas | BSD-3-Clause | https://pandas.pydata.org |
| scikit-learn | BSD-3-Clause | https://scikit-learn.org |
| XGBoost | Apache-2.0 | https://xgboost.ai |
| PyTorch | BSD-3-Clause | https://pytorch.org |
| ONNX | Apache-2.0 | https://onnx.ai |
| ONNX Runtime | MIT | https://onnxruntime.ai |
| PyYAML | MIT | https://pyyaml.org |

## UP-Fall dataset

Repository 不包含或重新散布 UP-Fall 真實受試者資料。資料集頁面提供公開存取與
引用論文，但本專案未找到一份明確適用於 dataset files 的開放授權。Sensors
論文的 CC BY 4.0 不會被自動延伸解讀到 dataset files。模型 artifacts 是研究產物；
下游公開發行前仍應由發行者確認資料集條款。必須引用：

Martínez-Villaseñor, L.; Ponce, H.; Brieva, J.; Moya-Albor, E.; Núñez-Martínez, J.;
Peñafort-Asturiano, C. *UP-Fall Detection Dataset: A Multimodal Approach*. Sensors 2019,
19(9), 1988. https://doi.org/10.3390/s19091988
