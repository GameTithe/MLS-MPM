# MPM 탄성 및 유체 시뮬레이션 (Taichi)

이 프로젝트는 Taichi 라이브러리를 이용해 탄성체와 유체 시뮬레이션을 구현한 예제입니다. Material Point Method(MPM)를 기반으로 하여 탄성체와 유체의 움직임을 시각적으로 표현합니다.

## 🎥 데모
![MLS-MPM](video/elastic.gif) ![MLS-MPM](video/water.gif)

## 🚀 주요 특징
- **탄성체(Elastic)** 및 **유체(Fluid)** 시뮬레이션 지원
- Taichi의 GPU 가속 지원 (`cuda`, `vulkan`, `cpu` 등) 

## 🛠️ 환경 설정
```bash
pip install taichi numpy
```

**주의**: GPU 사용 시 CUDA Toolkit이 설치되어 있어야 합니다.

## 📁 파일 구조
- `main.py`: 메인 시뮬레이션 코드

## 🧑‍💻 실행 방법
```bash
python main.py
```

## ⚙️ 시뮬레이션 옵션 변경하기
### 탄성체 시뮬레이션 활성화

#### Elastic
```python

def simulation():
 P2G()  # 주석 해제
 # P2G_1()
 # P2G_2()
```
#### Water 
```python
def simulation():
 #P2G()   
 P2G_1()
 P2G_2()
``` 

## 📚 참고 자료
- [MPM Course Notes](https://www.seas.upenn.edu/~cffjiang/mpmcourse/mpmcourse.pdf)
- [Taichi 공식 문서](https://docs.taichi.graphics/)

