# 고객 이탈(Attrition) 분석 EDA 보고서

이 프로젝트는 IBM HR Analytics Employee Attrition & Performance 데이터셋을 기반으로 고객 이탈 여부(Attrition)를 중심으로 한 탐색적 데이터 분석(EDA) 결과를 제공합니다.

---

## 데이터 개요

- 데이터 크기: 1470 rows × 35 columns  
- 분석 대상: 정규직 직원의 퇴사 여부
- 목적: 이탈과 관련된 주요 변수 파악 및 인사이트 도출

---

## 1. 기본 정보 분석

- 범주형 변수: Gender, JobRole, MaritalStatus, BusinessTravel 등  
- 수치형 변수: Age, MonthlyIncome, TotalWorkingYears 등  
- 결측치 없음  
- 중복 데이터 없음

---

## 2. 수치형 변수 통계 요약

| 변수명              | 평균 | 최소 | 최대 |
|---------------------|------|------|------|
| Age                 | 36.9 | 18   | 60   |
| MonthlyIncome       | 6500 | 1009 | 19999 |
| YearsAtCompany      | 7.0  | 0    | 40   |
| TotalWorkingYears   | 11.2 | 0    | 40   |

---

## 3. 범주형 변수 분포 및 이탈률

범주형 변수들 중 몇 가지는 이탈 여부(Attrition)와 명확한 차이를 보였습니다.

### OverTime (야근 여부)
- OverTime = Yes: 이탈률 **30.5%**
- OverTime = No: 이탈률 **8.0%**
- ➤ 야근이 많은 직원일수록 이탈 가능성이 높음

### JobRole (직무)
- Sales Executive: 이탈률 **20.6%**
- Research Scientist: 이탈률 **8.3%**
- ➤ 직무에 따라 이탈률 차이가 크며, 영업직의 이탈률이 높음

### BusinessTravel (출장 빈도)
- Travel_Frequently 그룹에서 이탈률이 가장 높음
- ➤ 잦은 출장은 이탈을 유발할 수 있는 요소

**요약**:  
근무 형태(야근, 출장)나 직무 특성은 이탈률과 밀접하게 관련되어 있으며, 이직 방지 전략 수립 시 중요한 고려 요소입니다.


---

## 4. 이탈 그룹별 수치형 변수 평균 비교

| 변수명            | 비이탈자 평균 | 이탈자 평균 | 차이 |
|-------------------|---------------|-------------|------|
| MonthlyIncome     | 6832          | 4787        | -2045 |
| YearsAtCompany    | 7.4           | 5.1         | -2.3  |
| TotalWorkingYears | 11.8          | 8.2         | -3.6  |

이탈자는 수입과 경력이 상대적으로 낮은 경향을 보임

---

## 5. 수치형 변수 간 상관관계

- MonthlyIncome ↔ TotalWorkingYears: 0.77  
- YearsAtCompany ↔ YearsWithCurrManager: 0.77  
- Attrition과의 직접 상관관계는 낮음 (범주형 변수 분석이 중요)

---

## 6. 파생변수 생성

- AgeGroup: 18-25, 26-35, 36-45, 46-55, 56+
- LogIncome: 월급 로그 변환
- JobChangeRate: NumCompaniesWorked / (TotalWorkingYears + 1)

---

## 7. 파생변수 기반 이탈률 분석

- 26-35세 그룹에서 이탈률이 상대적으로 높음
- JobChangeRate가 높은 경우 이탈 가능성 증가

---

## 8. 범주형 변수별 이탈률 시각화

- OverTime, JobRole, Department 등에서 이탈률 차이가 시각적으로 뚜렷하게 드러남
- 일부 직무(영업/연구)는 명확히 이탈 패턴이 존재

---

## 9. 변수 중요도 분석 (Random Forest 기반)

상위 5개 중요 변수:

1. OverTime  
2. MonthlyIncome  
3. JobLevel  
4. TotalWorkingYears  
5. JobRole  

---

## 10. 결론 및 다음 단계

이번 EDA를 통해 고객 이탈(Attrition)에 영향을 주는 주요 요인들을 다음과 같이 확인할 수 있었습니다.

### 주요 인사이트

- **야근 여부(OverTime)**: 야근을 하는 직원은 그렇지 않은 직원보다 이탈률이 3배 이상 높았습니다.
- **직무(JobRole)**: Sales Executive, Human Resources 등 일부 직무에서 이탈률이 유의하게 높게 나타났습니다.
- **수입(MonthlyIncome)**: 이탈자는 평균적으로 비이탈자보다 월급이 약 2,000달러 이상 낮은 경향이 있었습니다.
- **경력(YearsAtCompany / TotalWorkingYears)**: 경력이 짧을수록 이탈 가능성이 높았으며, 전체 경력 대비 현재 회사 재직기간이 짧은 경우 이탈 위험이 큽니다.

### 이탈 위험군 특성 요약

- **특징**:
  - OverTime = Yes
  - JobRole = Sales Executive, HR 등
  - MonthlyIncome 하위 그룹
  - TotalWorkingYears < 평균 (약 11년)

이러한 조건이 복합적으로 작용하는 직원은 **이탈 가능성이 높고**, 이들을 조기 식별하는 것이 중요합니다.

---
