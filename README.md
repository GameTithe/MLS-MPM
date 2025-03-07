MPM 탄성 및 유체 시뮬레이션 (Taichi)

이 프로젝트는 Taichi 라이브러리를 이용해 탄성체와 유체 시뮬레이션을 구현한 예제입니다. Material Point Method(MPM)를 기반으로 하여 탄성체와 유체의 움직임을 시각적으로 표현합니다.

🎥 데모

![MLS-MPM](video/elastic.gif)
![MLS-MPM](video/water.gif)


실행 시 시뮬레이션 입자들이 화면에서 움직이며 유체 또는 탄성체 거동을 시각화하여 확인할 수 있습니다.
 
🚀 주요 특징

탄성체(Elastic) 및 유체(Fluid) 시뮬레이션 지원

Taichi의 GPU 가속 지원 (cuda, vulkan, cpu 등)

GGUI 기반의 실시간 시각화

🛠️ 환경 설정

pip install taichi numpy

주의: GPU 사용 시 CUDA Toolkit이 설치되어 있어야 합니다.

📁 파일 구조

main.py: 메인 시뮬레이션 코드

🧑‍💻 실행 방법

python main.py

⚙️ 시뮬레이션 옵션 변경하기

탄성체 시뮬레이션 활성화

탄성체를 테스트하고자 한다면 simulation() 함수 내에서:

P2G()  # 주석 해제
# P2G_1()
# P2G_2()

위와 같이 수정하면 탄성체 시뮬레이션이 동작합니다.

유체 시뮬레이션 활성화 (기본값)

기본으로 유체 시뮬레이션이 활성화되어 있습니다.

# P2G()  # 주석 처리
P2G_1()
P2G_2()

📚 참고 자료

MPM Course Notes

Taichi 공식 문서
 
