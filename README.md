#🌀 2D 유체 시뮬레이션

#📌 소개

이 프로젝트는 2D Navier-Stokes 방정식을 이용한 유체 시뮬레이션을 구현한 Python 코드입니다.
유체의 흐름을 시각적으로 표현하며, 압력 계산, 속도 필드 보정, 외력 적용 등의 핵심 요소를 포함합니다.
    
#4️⃣ 유체 시뮬레이션 단계

힘 적용: 외력을 속도 필드에 추가

비선형 대류 (자기운반): 속도 필드를 자기 자신으로 이동

점성 확산: 점성을 고려한 확산 연산 적용

압력 보정: 압력을 계산하여 발산 없는 속도장을 생성
   
