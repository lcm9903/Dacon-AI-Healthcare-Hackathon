# 유전자 변이 기반 암 아형(SUBCLASS) 분류

> 데이콘 **[초격차] AI 헬스케어 7기 해커톤** 최종 제출 코드
> 환자별 유전자 변이 정보(4,384개 유전자)로 26개 암종(SUBCLASS)을 분류합니다.

| 항목 | 내용 |
|---|---|
| 평가 지표 | Macro F1 |
| 최종 모델 | LightGBM (시드 3개 앙상블) + 로지스틱 회귀 소프트 보팅 (LR 가중치 0.7) |
| 교차검증 | StratifiedGroupKFold 5-Fold (변이 프로필이 같은 환자는 같은 fold) |
| OOF Macro F1 | **0.5098** (LightGBM 0.4768 / 로지스틱 회귀 0.4868) |
| 외부 데이터·사전 학습 모델 | 사용하지 않음 |
| 실행 환경 | Google Colab (CPU) |

---

## 📁 폴더 구조

```
.
├── README.md
├── <노트북 파일>.ipynb          # 전처리 → 학습 → 예측 → 제출 파일 생성까지 한 번에
└── data/
    ├── train.csv                 # 입력 (6,201명 × 4,386열)
    ├── test.csv                  # 입력 (2,546명 × 4,385열)
    ├── sample_submission.csv     # 입력
    ├── submission_lightgbm_final_dup.csv   # 출력: 제출 파일
    └── model/                    # 출력: 추론에 쓴 모델 파일
        ├── lgbm_seed42.txt
        ├── lgbm_seed7.txt
        ├── lgbm_seed2026.txt
        └── preprocess_and_lr.pkl
```

## ▶️ 실행 방법

1. `train.csv`, `test.csv`, `sample_submission.csv`를 데이터 폴더에 넣습니다.
2. **1장 셀**의 `DATA_DIR`을 데이터 폴더 경로로 맞춥니다.
   ```python
   DATA_DIR = './data'    # 현재 노트북 값: '/content/sample_data'
   ```
3. 노트북을 위에서부터 순서대로 전체 실행합니다. 처음 한 번은 0장에서 라이브러리를 설치합니다.
   ```bash
   pip install lightgbm koreanize-matplotlib
   ```
4. 실행이 끝나면 `DATA_DIR`에 제출 파일, `DATA_DIR/model/`에 모델 파일이 저장됩니다.

> ⏱ 소요 시간: Colab CPU 기준 fold당 약 3~4분, 전체 약 30분 이상 (최종 학습은 시드 수만큼 반복)

## 🧪 개발 환경

| 라이브러리 | 버전 |
|---|---|
| Python | 3.13.18 |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| scipy | 1.16.3 |
| scikit-learn | 1.6.1 |
| lightgbm | 4.6.0 |
| matplotlib | 3.10.0 |

OS: Linux-6.6.159+-x86_64 (Google Colab) · 파일 인코딩 UTF-8 · 시드 `42`

---

## 🔍 파이프라인

```
원본 CSV ─▶ 결측치 처리(WT) ─▶ 변이 파싱(행 단위) ─▶ 피처 생성 ─▶ 그룹 5-Fold 학습
                                                            │
                         제출 파일 ◀─ 중복 프로필 후처리 ◀─ 전체 Train 재학습 · Test 예측
```

### 1. 전처리
- **결측치**: 이 데이터에서 "변이 없음"은 `WT`(Wild Type)로 표시됩니다. Test에만 있는 NaN(237칸, 128명)은 Test 통계가 아닌 **고정값 `WT`**로 채웁니다.
- **희소성**: 전체 칸의 99.19%가 `WT`인 매우 희소한 데이터입니다.
- **변이 파싱**: 한 칸에 공백으로 구분된 여러 변이(`Q369* I368N`)를 하나씩 나눠 세 종류로 분류합니다.

| 종류 | 판단 기준 | 예시 |
|---|---|---|
| 침묵(silent) | 앞뒤 아미노산이 같음 | `R895R` |
| 절단(truncating) | `*`(넌센스) 또는 `fs`(프레임시프트) | `Q369*`, `K120fs` |
| 미스센스(missense) | 그 외 아미노산 치환 | `V600E` |

### 2. 피처 생성

| 단계 | 피처 | 설명 |
|---|---|---|
| 기본 | 변이 토큰 | `TP53`, `TP53__R273H`, `TP53__TRUNC` 희소 행렬 (트리: 0/1, 선형: TF-IDF) |
| 기본 | 행 단위 통계 | 변이 유전자 수, 총 변이 수, 종류별 개수·비율 |
| 1차 | 변이 부담·핫스팟 | 미스센스 비율, 다중 변이 유전자 수, 과변이 여부(상위 10%), 핫스팟 변이 수 |
| 2차 | 변이 패턴 | 아미노산 치환(`R>H`), 코돈 위치 토큰(`IDH1__R132`), 넌센스·프레임시프트 분리, 절단 경향 유전자, 반복 변이 코돈 수 |
| 3차 | 아미노산 성질 | 치환 상위 100종 비율, 성질 그룹 변화(소수성·극성·양전하·음전하·글리신·프롤린, 최대 36종) |
| 최종 | 나이브 베이즈 점수 | 환자의 변이 토큰 전체가 가리키는 암종 26개 확률 (중첩 CV로 누수 방지) |
| 최종 | LightGBM 전용 | 유전자 쌍 동시 변이(`pair:IDH1&ATRX`, 상위 200쌍), 변이 부담 × 변이 종류 조합 통계 5개 |

각 피처 단계는 1장의 스위치(`USE_NEW_FEATURES`, `USE_MORE_FEATURES`, `USE_AA_GROUPS`, `USE_NB_FEATURES`, `USE_PAIR_FEATURES`, `USE_COMBO_FEATURES`, `USE_POS_BINS`)로 켜고 끌 수 있습니다.

### 3. 교차검증
같은 환자가 두 코호트에 모두 등록된 경우(KIPAN↔KIRC, GBMLGG↔LGG)가 있어 변이 프로필이 똑같은 행이 1,016개 있습니다. 행 해시로 그룹을 만들어 **StratifiedGroupKFold**로 같은 프로필을 같은 fold에 묶어 검증 점수가 부풀려지지 않게 했습니다.

### 4. 모델

**LightGBM**

```python
objective='multiclass', learning_rate=0.05, num_leaves=15, min_child_samples=5,
colsample_bytree=0.3, subsample=0.8, reg_lambda=1.0, class_weight='balanced'
# 검증 fold multi_logloss 기준 조기 종료(100), 최종 반복 수 = fold 평균 × 1.1 (174)
```

**로지스틱 회귀**: `C=2`, `class_weight='balanced'`, TF-IDF 피처

**앙상블**: OOF Macro F1이 가장 높은 가중치(0.0~1.0, 0.1 간격 탐색)를 선택 → `0.3 × LightGBM + 0.7 × LR`

### 5. 최종 학습 및 예측
전체 Train으로 다시 학습합니다. LightGBM은 시드 `[42, 7, 2026]` 3개로 각각 학습해 확률을 평균합니다.

### 6. 후처리 — 중복 프로필 규칙
Train의 중복 프로필은 거의 모두 KIPAN+KIRC, GBMLGG+LGG 조합입니다(KIPAN은 KIRC를, GBMLGG는 LGG를 포함하는 코호트). 그래서 Test 환자의 프로필이 Train의 KIRC 환자와 똑같으면 짝 라벨인 KIPAN으로 예측합니다.
- 짝 라벨 목록은 Train 안의 중복 쌍(20쌍 이상)만으로 만듭니다.
- Test에서 Train과 프로필이 같은 266명 중 222명에 적용, 219명의 예측이 바뀌었습니다.
- OOF 시뮬레이션에서 Macro F1 약 **+0.046** 개선을 확인했습니다.

---

## 📊 결과

### 리더보드 기록

| 버전 | Public 점수 |
|---|---|
| 분포 보정값 제거 | 0.3246 |
| + 1차 피처 | 0.3355 |
| + 2차 피처 | 0.3727 |
| + 3차 피처 | 0.3848 |
| 최종 (나이브 베이즈 + LightGBM 전용 피처 + 시드 앙상블 + 후처리) | 이 노트북 |

### OOF 성능

| 모델 | OOF Macro F1 | fold 평균 | fold 표준편차 |
|---|---|---|---|
| LightGBM 단독 | 0.4768 | 0.4778 | 0.0177 |
| 로지스틱 회귀 단독 | 0.4868 | 0.4896 | 0.0155 |
| **앙상블 (LR 0.7)** | **0.5098** | **0.5125** | 0.0220 |

- 상위 1 / 3 / 5개 예측 안에 정답이 들어갈 비율: 51.6% / 77.4% / 86.4%
- 암종별 F1: SKCM(0.897)·ACC(0.875)·COAD(0.779)는 높고, THYM(0.143)·SARC(0.234)·KIRC(0.333)는 낮음
- 가장 많이 헷갈린 쌍: KIRC↔KIPAN, GBMLGG↔LGG (중복 환자 때문) / BRCA→PRAD, BRCA→OV

### 분석에서 얻은 것
- **이론상 상한**: 프로필이 똑같은데 라벨이 다른 환자 때문에 줄일 수 없는 오류가 511명(8.2%) 있습니다. 모든 변이가 `WT`인 환자 94명은 라벨이 15종으로 나뉩니다.
- **변이량에 따른 성능**: 변이 유전자가 적을수록(0~5개) 예측이 어렵고, 51개 이상이면 정확도가 70% 이상으로 오릅니다.
- **시도했지만 채택하지 않은 것**: 나이브 베이즈 변이량 정규화(`sqrt`, `mean`), 단백질 위치 구간 토큰(`USE_POS_BINS`), Test 예측 분포 보정

---

## 🛡 Data Leakage 방지

- 파싱·통계·프로필 해시는 **각 행 안에서만** 계산합니다.
- 결측치는 Test 통계가 아닌 **고정값 `WT`**로 채웁니다.
- Vectorizer 어휘, LabelEncoder, 과변이 기준값, 핫스팟·코돈·유전자 목록, 유전자 쌍, 나이브 베이즈, 모델은 모두 **Train(학습 fold)으로만** 학습하고 Test에는 적용만 합니다.
- 나이브 베이즈 점수는 학습 fold 안에서 다시 5-Fold로 나눈 **중첩 CV**로 만들어 자기 정답이 피처에 섞이지 않게 했습니다.
- 중복 프로필 규칙의 짝 라벨 목록은 Train으로만 만들고, Test는 행 해시만 대조합니다.
- Test로 EDA를 하지 않으며, Test 예측 분포에 맞춘 확률 보정도 하지 않습니다.

## 💾 모델 파일

| 파일 | 내용 |
|---|---|
| `lgbm_seed{42,7,2026}.txt` | 최종 LightGBM 모델 (시드별, LightGBM 텍스트 형식, 각 8.5MB) |
| `preprocess_and_lr.pkl` | 트리용·선형용·LightGBM 전용 특징 변환기, 로지스틱 회귀 모델, LabelEncoder, 앙상블 가중치, 설정값 (9.8MB) |

나이브 베이즈 점수는 Train으로 다시 계산되므로 별도 파일 없이 같은 코드로 재현됩니다.

## Contributors

<a href="https://github.com/lcm9903">
  <img src="https://github.com/lcm9903.png" width="40" height="40" alt="lcm9903" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/yeohj1226">
  <img src="https://github.com/yeohj1226.png" width="40" height="40" alt="yeohj1226" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/redisfree">
  <img src="https://github.com/redisfree.png" width="40" height="40" alt="redisfree" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/wlgml2244-jpg">
  <img src="https://github.com/wlgml2244-jpg.png" width="40" height="40" alt="wlgml2244-jpg" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/claude">
  <img src="https://github.com/claude.png" width="40" height="40" alt="claude" style="border-radius: 50%;"/>
</a>
