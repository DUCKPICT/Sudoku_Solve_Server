# Sudoku_Solve_Server
- 스도쿠 데이터를 가지고 AI 모델을 구축하고 그 모델을 통해서 스도쿠해결

##Program Version
- Ubuntu Version : 22.04

- TensorFlow Version : 2.18.0
- Install Code(pip install tensorflow==2.18.0)
- Re-Install Code(pip uninstall tensorflow) -> (pip install tensorflow)

- Keras Version : 3.6.0
- Install Code(pip install keras==3.6.0)
- Re-Install Code(pip uninstall keras) -> (pip install keras==3.6.0)
- Python Version : 3.10.12

##Download sudoku.csv
- https://www.kaggle.com/datasets/rohanrao/sudoku?resource=download
- 다운로드후 content폴더속에 .csv파일 복사

##Change Code Part
- app.py
- Line 26. Change to Your Location부분 수정하기
- (Training_Sudoku.py 실행이후 폴더에 생기는 best_weights.keras의 Location)

- Training_Sudoku.py
- Line 15. Change to Your Location부분 수정하기
- (Content 폴더속 sudoku.csv파일의 Location)

##Training HyperParameter
- Training_Sudoku.py
- Line 99. epochs 수정 학습 반복횟수

실행 방법
Download File Locatiton 이동후 Training_Sudoku.py 실행(python3 Training_Sudoku.py or python Training_Sudoku.py)
학습 완료후 app.py실행(python3 app.py or python app.py)

