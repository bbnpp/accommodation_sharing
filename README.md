# accommodation_sharing

#### Motivation
- 숙박 공유 플랫폼에서 호스트의 가격 설정은 숙소의 인기에 가장 큰 영향을 미치는 요소임  
- 호스트가 책정한 숙박 요금이 적정하지 못할 경우 초과수요 또는 초과공급을 유발함  
- 숙박 요금을 재조정함으로써 모든 플랫폼 참여자가 이득을 볼 수 있음  
- 숙소의 특성 데이터를 활용해 적정 가격을 산정하여 호스트가 참고할 수 있게 하고자 함  

#### Define problematic accommodations
- 일정 기간 내 각 숙소의 예약 일수로 Gaussian mixture models 학습
- Fitted GMM 을 이용한 Outlier detection 수행
- Outliers를 가격 조정이 필요한 숙소로 정의

#### Features
- 개별 숙소의 메타데이터
- Uber H3 Hexagon을 활용한 Neighborhood information extraction 수행

#### Predict proper price for accommodations
