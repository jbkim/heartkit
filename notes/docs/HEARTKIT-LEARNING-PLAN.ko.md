# heartKIT 구조 학습 & 온디바이스(TFLM) 실습 계획

> 목표: heartKIT(https://ambiqai.github.io/heartkit/)의 구조와 전체 파이프라인을 분석하고,
> Apollo510B EVB(+ MAX86150 센서)에서 경량화(quantized/TFLM) 모델을 실제로 돌리는 법을 학습한다.
> 코드 수정/기능 추가가 목적이 아니라 **학습 목적**.
> 2026-08-28 기준 작성.

---

## 0. 진행 상황 요약 (이 문서 작성 시점까지)

- macOS(Apple Silicon)에서 `tensorflow-metal` 플러그인 충돌로 `heartkit`이 실행되지 않던 문제 해결
  → [SETUP-TROUBLESHOOTING.ko.md](SETUP-TROUBLESHOOTING.ko.md), [configs/smoke-test.json](../configs/smoke-test.json)
- 합성 데이터(`ecg-synthetic`)로 segmentation 태스크의 `train → evaluate → export` 파이프라인을 1분 스모크 테스트로 검증 완료 (`results/smoke-test/`)
- Ambiq Model Zoo의 사전학습 rhythm(부정맥) 모델(`arr-2-eff-sm`, EfficientNetV2)을 실데이터(ptbxl 3.7G, lsad 6.1G)로 평가 성공
  - 전체 test set: ACC=0.9841, F1=0.9853
  - confidence threshold(75%) 적용 시: ACC=0.9973, F1=0.9973
  - 평가 전용 설정 [configs/arr-2-eff-sm-eval.json](../configs/arr-2-eff-sm-eval.json), [configs/download-arr-2-eff-sm.json](../configs/download-arr-2-eff-sm.json) 작성 (원본 zoo 설정 `configs/arr-2-eff-sm.json`은 AmbiqAI 배포본 그대로 보존 — train 시 val_file 캐시 재현성 때문)
- 보유 하드웨어 확인: **Apollo510B EVB + MAX86150(모듈 소켓 장착)**

---

## 1. 학습 로드맵 (레이어 A→D)

아래에서 위로: 코드 구조 → 파이프라인 실행 → 경량화 산출물 해석 → 온디바이스 연동.

### 레이어 A. 정적 구조 분석 (코드 읽기, 실행 없음)

- [x] `heartkit/tasks/__init__.py` — `TaskFactory`는 `helia.utils.create_factory`로 만든 싱글턴. 7개 태스크(segmentation/rhythm/beat/denoise/diagnostic/foundation/translate)를 문자열 키로 등록
- [x] `heartkit/tasks/task.py` — `HKTask` 추상 베이스: `download()`는 기본 구현 제공(각 데이터셋의 `.download()` 호출), `train`/`evaluate`/`export`/`demo`는 `NotImplementedError` — 각 태스크가 반드시 구현해야 하는 4개 훅
- [x] `heartkit/backends/{backend,pc,evb}.py` — `HKInferenceBackend` 추상 클래스(`open/close/set_inputs/perform_inference/get_outputs` 5개 메서드)를 `PcBackend`(로컬 Keras/TFLite)와 `EvbBackend`(eRPC 시리얼)가 각각 구현. `demo.py`가 `BackendFactory.get(params.backend)`로 다형적으로 선택 — 이게 BYOB(Bring-Your-Own-Backend) 확장점
- [x] `heartkit/tasks/segmentation/__init__.py` — `SegmentationTask(HKTask)`는 각 메서드를 같은 패키지의 `train.py`/`evaluate.py`/`export.py`/`demo.py` 모듈 함수로 그대로 위임하는 얇은 어댑터일 뿐 (다른 6개 태스크도 동일 패턴일 것으로 추정, 미검증)
- [x] 모델 팩토리는 외부 패키지 `helia-edge`로 이관됨 — TCN/U-Net/U-NeXt/EfficientNetV2/MobileOne/ResNet/Conformer/MetaFormer/TSMixer 등록

**📌 문서-코드 불일치 추가 발견**: `docs/modes/demo.md`는 "HTML report will be saved to `${job_dir}/report.html`"라고 적혀 있지만, 실제 `segmentation/demo.py:276`은 `params.job_dir / "demo.html"`에 저장한다 (파일명이 다름 — `report.html`이 아니라 `demo.html`/`demo.png`).

### 레이어 B. 모드별 파이프라인 실행

- [x] download / train / evaluate / export (segmentation, smoke-test)
- [x] evaluate (rhythm, zoo 모델, 실데이터)
- [x] **demo** 모드 — `backend: "pc"`로 실행 성공 (`heartkit -m demo -t segmentation -c ./notes/configs/smoke-test.json`) → `results/smoke-test/demo.html`, `demo.png` 생성 확인. R-peak 검출, HRV(시간/주파수영역), Poincare plot까지 한 번에 리포트됨. (smoke-test 모델은 2 epoch만 학습해 결과 자체는 무작위 수준 — 파이프라인 동작 확인용)

### 레이어 C. 경량화 산출물 해석

- [x] `quantization.format` 비교 (smoke-test 모델, 유효값은 `FP32`/`FP16`/`INT8`/`INT16X8` — 소문자 `int8`이 아니라 대문자 StrEnum임을 `helia_edge.converters.tflite.converter.QuantizationType`에서 확인):

  | format | model.tflite 크기 | TFL ACC | TFL F1 |
  | --- | --- | --- | --- |
  | FP32 | 12,028 B | 0.4549 | 0.3567 |
  | FP16 | 10,960 B | 0.4508 | 0.3521 |
  | INT8 | 10,704 B | 0.4751 | 0.3696 |

  smoke-test 모델(2개 conv 블록, 16~24 필터)은 너무 작아서 양자화 효과가 뚜렷하지 않음 — tflite 파일 크기의 상당 부분이 그래프 구조/메타데이터 오버헤드라 가중치 압축 효과가 묻힘. **더 큰 모델에서 재확인 필요** (아래 참고).
- [x] 실제 규모 모델(`arr-2-eff-sm`, 18K 파라미터)로 FP32 vs INT8 재비교:

  | format | model.tflite 크기 | TFL ACC | TFL F1 |
  | --- | --- | --- | --- |
  | FP32 | 90,456 B | 0.9849 | 0.9860 |
  | INT8 | 63,800 B | 0.9823 | 0.9835 |

  크기는 약 **29% 감소**(90.4KB→63.8KB), 정확도는 F1 기준 **0.0025 하락**만 발생. 4배 축소를 기대했다면 의외로 작은 폭인데, INT8 변환이 가중치뿐 아니라 활성화/그래프 메타데이터까지 포함한 전체 flatbuffer 크기 기준이라 순수 가중치 압축률(이론상 4배)이 그대로 반영되지 않기 때문 — "INT8 = 무조건 1/4 크기"라는 통념과 실제 수치가 다르다는 걸 직접 확인한 게 이 실습의 핵심 소득.
- [x] 파라미터 수 대조: rhythm evaluate 로그의 `Total params: 53,615` 중 Trainable 17,562 + Non-trainable 928 ≈ **18.5K** ↔ `docs/zoo/index.md`의 ARR-2-EFF-SM 문서화 값 **18K**와 일치 (나머지 35,125는 옵티마이저 상태값이라 모델 크기에 포함 안 됨)
- [x] FLOPS 대조: export 로그 `Model requires 1.23 MFLOPS` ↔ 문서화 값 **1.2M**과 일치 (FP32/INT8 둘 다 동일 — FLOPS는 양자화 전 Keras 그래프 기준으로 계산되므로 quantization format과 무관)
- [x] `model_buffer.h`(TFLM C 헤더) 구조 확인: `const unsigned char g_smoke_model[] = { 0x20, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ... }` — 4번째 바이트부터 `TFL3`(TensorFlow Lite FlatBuffer 매직 넘버)로 시작, `.tflite` 파일을 바이트 그대로 담은 배열. `const unsigned int g_smoke_model_len = 10704;` 로 길이 상수 동봉(INT8 `model.tflite`의 실제 바이트 수와 정확히 일치 확인). `#ifndef __G_..._H` include guard만 있고 `alignas` 등 정렬 지정은 없음 — 정렬은 펌웨어 쪽에서 처리(예: neuralSPOT `evb/src/main.cc`의 `alignas(16) unsigned char modelBuffer[...]`로 복사해 넣는 방식)

### 레이어 D. 온디바이스 연동 (Apollo510B 보유 → 실물 실습 경로)

아래 "2. 펌웨어 아키텍처 관련 발견" 참고 후 진행.

- [x] neuralSPOT 조사 → `rpc_pc_to_evb`는 해당 없음 확인, `AmbiqAI/tileio-demos`가 실제 대상임을 특정
- [ ] `tileio-demos` 클론 및 Apollo510B 빌드 타깃 시도
- [ ] MAX86150 배선 확인 (Apollo510B EVB 기준 — 구식 문서의 Apollo4 핀맵 재사용 불가)
- [ ] 빌드/플래싱 성공 시 Tileio 앱으로 실물 데모 확인 (heartkit CLI의 `backend: "evb"` 경로가 아니라 Tileio 앱 기반 경로로 대체됨에 유의)

### 레이어 E. 선택 (시간 되면)

- [ ] beat/denoise 등 다른 태스크로 스모크 테스트 1회씩 → 태스크 간 config 차이(입력 shape, 클래스 수, 모델 구조) 비교

---

## 2. 펌웨어 아키텍처 관련 발견

### 문제: 공식 가이드 문서가 낡음

`docs/guides/evb-setup.md`, `docs/guides/rhythm-demo.md`는 **2024-03-13 이후 업데이트가 없음**. 이 문서들은 저장소 루트에 `./evb/` 디렉터리(Makefile 기반 펌웨어 프로젝트)가 있다고 전제하고 `make -C ./evb`, `make -C ./evb deploy` 같은 명령을 안내하지만, 이 디렉터리는 로컬 클론과 GitHub 업스트림(`AmbiqAI/heartkit` main) 어디에도 존재하지 않는다 (확인: `gh api repos/AmbiqAI/heartkit/contents/`).

### 원인: 2024-04-24 아키텍처 전환

커밋 `e4bfaab "feat: Remove evb code. feat: Add eRPC python code."` 에서 EVB 펌웨어를 **저장소에서 완전히 제거**하고 **eRPC 기반 통신**으로 전환했다. 즉 예전에는 태스크/모델마다 펌웨어를 재컴파일해야 했지만, 지금은:

1. EVB에는 **범용(generic) eRPC 리스너 펌웨어**를 한 번 플래싱해두고
2. PC(`heartkit`)가 시리얼로 모델 + 입력 데이터를 그때그때 전송해 추론시키는 구조

`heartkit/backends/evb.py`의 `EvbBackend`가 이 흐름의 PC 측 클라이언트다 (`open()` 시 시리얼 포트 스캔 → 연결 → `send_model()`으로 모델 전송, `RpcCommands`: SEND_MODEL/SEND_INPUT/FETCH_OUTPUT/FETCH_STATUS/PERFORM_INFERENCE).

### (1차 가설, 정정됨) `neuralSPOT`의 `rpc_pc_to_evb`는 정답이 아니었음

처음에는 eRPC 인터페이스명(`GenericDataOperations_PcToEvb`)을 GitHub 코드 검색으로 추적해 `AmbiqAI/neuralSPOT`의 `apps/examples/rpc_pc_to_evb/`가 그 펌웨어라고 추정했다. 하지만 실제로 그 예제는 "dataBlock을 보내고 2배로 곱해서 돌려받는" **범용 전송 계층 데모**일 뿐, heartkit이 기대하는 `SEND_MODEL`/`SEND_INPUT`/`PERFORM_INFERENCE` 같은 애플리케이션 레벨 명령을 구현하고 있지 않다 (`grep`으로 확인 — neuralSPOT 어디에도 이 명령어 이름이 없음).

### 원래 구현 위치(2024-04 이전)와 현재 위치

과거 `evb/` 삭제 커밋(`e4bfaab`) 직전 버전을 git 히스토리에서 열어보면, `evb/src/main.cc`가 정확히 이 프로토콜(SEND_MODEL 등)을 구현하고 있었다: neuralSPOT의 범용 RPC 전송(`ns_rpc_generic_data`) 위에 **heartkit이 직접 작성한 TFLM 추론 서버 콜백**(`ns_rpc_data_to_evb_cb` 등)을 얹은 구조였다. 이 앱 코드는 Apollo4 Plus + AmbiqSuite R4.4.1 기준으로 작성돼 있었고, 삭제된 뒤 heartkit/neuralSPOT 어디에도 재게시되지 않았다.

GitHub 전역 코드 검색(`ns_rpc_data_to_evb_cb`)으로 이 콜백명을 그대로 쓰는 **현재 유지보수 중인 버전**을 찾았다:

- **`AmbiqAI/tileio-demos`** 저장소 — [Tileio](https://github.com/AmbiqAI/tileio)(Ambiq의 모바일/웹 대시보드 앱)와 짝을 이루는 데모 모음. `heartkit/` 디렉터리가 바로 이 EVB 펌웨어의 현재 버전이다.
  - 아키텍처: Denoise → Segmentation → (HRV / 4-class Rhythm) → **BLE/USB로 Tileio 앱에 스트리밍** — 예전 CLI 기반 `heartkit -m demo backend:"evb"` + SWO 출력 방식이 아니라, **모바일/웹 대시보드 앱으로 보는 방식**으로 완전히 바뀌었다.
  - 사용 모델: `assets/den-tcn-sm.json`(denoise), `assets/seg-4-tcn-sm.json`(segmentation), `assets/arr-4-eff-sm.json`(rhythm 4-class) — 우리가 다뤘던 `arr-2-eff-sm`(2-class)과는 다른 설정.

### ⚠️ 중요한 제약: Apollo510B는 공식 지원 목록에 없음

`tileio-demos/heartkit/README.md`의 **Supported Platforms**에 명시된 보드는 다음 3종뿐이다:

- `apollo4p_evb`
- `apollo4p_blue_kxr_evb`
- `apollo4l_blue_evb`

**모두 Apollo4 계열이며, 사용자가 보유한 Apollo510B(Apollo5 계열)는 목록에 없다.** `autogen_*.mk` 빌드 설정 파일도 `apollo4p_blue_kxr_evb`용 하나만 저장소에 있다. neuralSPOT 자체는 Apollo510B를 지원하지만(`neuralspot/ns-core/src/apollo510b/` 존재 확인), 이 heartkit 데모 앱이 Apollo510B용으로 빌드되도록 포팅되어 있는지는 **미검증 — 직접 `PLATFORM=apollo510b_evb`로 빌드를 시도해봐야 확인 가능**.

### 밝은 신호: `apollo510b_evb`는 neuralSPOT 빌드 시스템에 등록된 정식 플랫폼명

`neuralSPOT/make/neuralspot_config.mk:180`에 `ifeq ($(PLATFORM), apollo510b_evb)` 분기가 실제로 존재하고, `extern/AmbiqSuite/`에 Apollo5 계열 SDK(R5.1.0_rc27, R5.2.alpha.1.1)도 이미 벤더링되어 있다. 즉 **neuralSPOT 자체의 Apollo510B 지원은 실질적**이며, `tileio-demos/heartkit`가 미지원 보드 목록에 있는 것은 "아직 autogen 캐시 파일을 안 만들어서/공식 검증을 안 해서"일 가능성이 있다 — 실제로 안 될지는 빌드를 시도해봐야 확정.

### 로컬 툴체인 상태 (2026-08-28 확인)

- `arm-none-eabi-gcc` 있음, 그러나 **"GNU Tools for STM32 11.3"** (STM32CubeCLT에 딸려온 것) — heartkit/neuralSPOT이 요구하는 **공식 Arm GNU Toolchain ^12.2**와 다름. 버전 불일치로 빌드가 실패하거나 미묘한 문제가 생길 수 있어 공식 Arm GNU Toolchain 12.2 설치 권장.
- `JLinkExe` 있음 (`/usr/local/bin/JLinkExe`) — 버전은 별도 확인 필요 (README 요구사항: v7.92+).

### 실측 결과 (2026-08-28)

1. **`tileio-demos/heartkit` 자체 빌드는 실패함** — 원인은 툴체인이 아니라 **서브모듈 버전**. `tileio-demos`가 고정한 neuralSPOT 서브모듈 커밋(`e40475c5`, 2024-09-12)에는 `apollo510` 계열 보드 코드가 전혀 없음(grep 0건). Apollo510 지원은 그 이후 neuralSPOT에 추가된 기능.
2. 별도로 클론한 **최신 neuralSPOT(main HEAD, 2026-03-24)** 에서 `PLATFORM=apollo510b_evb`로 시도 → `am_bsp.h: No such file` 로 실패. 원인: AmbiqSuite 패키지의 실제 보드 폴더명은 `boards/apollo510_evb`(“b” 없음)인데 `apollo510b_evb`라는 PLATFORM 값이 존재하지 않는 보드로 해석됨.
3. **`PLATFORM=apollo510_evb`(“b” 없이)로 재시도 → 완전 성공.** TFLM(`helia_rt_v1_6_0`), CMSIS-DSP(`CMSISDSP-m55-gcc`), eRPC, AmbiqSuite R5.3.0(자동 선택) 전부 컴파일/링크 성공. `make nestall`로 `nest`(라이브러리+헤더+`autogen_apollo510_evb_arm-none-eabi.mk`) 생성까지 확인.
   → **결론: STM32CubeCLT의 GCC 11.3 툴체인도 문제없이 Apollo510B(neuralSPOT 표기로는 `apollo510_evb`)를 빌드할 수 있음이 실증됨.**

### 남은 과제: heartkit 데모 앱 자체의 이식

SDK(neuralSPOT) 자체는 Apollo510B를 완전히 지원하지만, `tileio-demos/heartkit`의 애플리케이션 소스(`src/main.cc`, `src/tflm.cc` 등)는 **1.5년 전 neuralSPOT API 기준으로 작성**되어 있다. 이 소스를 최신 nest(위 3번에서 생성한 것)에 연결해서 다시 빌드하면, 그 사이 API가 바뀐 지점에서 컴파일 에러가 날 가능성이 높다 — 이건 실제로 시도해봐야 범위를 알 수 있는 **별도의 이식 작업**이다.

### 확인이 더 필요한 부분

- `tileio-demos/heartkit` 소스를 새 nest(`apollo510_evb`)에 연결했을 때 실제로 컴파일되는지, 안 된다면 에러 범위(API 몇 곳만 손보면 되는지, 구조적으로 안 맞는지)
- Apollo510B EVB에서 MAX86150 배선 (구식 `evb-setup.md`의 J17/J11 핀맵은 Apollo4 EVB 기준 — Apollo510B 보드 실물의 실크스크린/핀아웃을 직접 확인 필요)
- Tileio 앱(iOS/웹) 없이 CLI만으로 결과를 볼 방법이 있는지, 아니면 이 경로는 Tileio 앱 사용이 필수인지

---

## 3. 다음 단계

> **결정 (2026-08-28)**: Apollo510B 실물 이식(레이어 D)은 상당한 임베디드 포팅 작업으로 확인되어 잠시 보류. **레이어 A(구조 분석) → B(demo PC 백엔드) → C(경량화 해석)를 먼저 진행**하기로 함. 아래 목록은 D로 돌아올 때의 재개 지점.


1. ~~`AmbiqAI/neuralSPOT` 클론~~ — 완료. `rpc_pc_to_evb`는 범용 전송 데모일 뿐 heartkit 프로토콜과 무관함을 확인, 폐기
2. `AmbiqAI/tileio-demos` 클론 (`--recurse-submodules`, `heartkit`/`neuralspot` 서브모듈 포함)
3. `tileio-demos/heartkit`를 `PLATFORM=apollo510b_evb`(정확한 이름은 `make` 도움말/neuralSPOT PLATFORM 목록에서 확인)로 빌드 시도 — 공식 미지원 보드이므로 실패 가능성 있음, 실패 시 에러 로그로 원인(드라이버 미포팅 vs 단순 설정 누락) 판별
4. 되면: 실제 플래싱 → Tileio 앱(iOS/웹)으로 대시보드 연결하여 실물 데모 확인
5. 안 되면: 대안 두 가지
   - (a) `backend: "pc"`로 데모까지만 학습 목적 달성하고, EVB 실습은 "코드 리딩 + 아키텍처 이해"로 대체 (레이어 D를 코드 스터디로 전환)
   - (b) `tileio-demos/heartkit`를 Apollo510B용으로 직접 포팅 시도 (별도의 상당한 임베디드 작업 — 이번 학습 목적 범위를 넘을 수 있으므로 착수 전 사용자와 범위 재확인)

---

## 4. 완료 기준

- [x] 7개 태스크 × 5개 모드의 공통 인터페이스를 소스코드 근거로 설명 가능 (`HKTask`/`HKInferenceBackend`/`TaskFactory`/`BackendFactory` 구조 확인, §1 레이어 A)
- [x] `pc` 백엔드로 demo 모드 1회 이상 성공 실행, 결과 해석 (실제 파일명은 `report.html`이 아니라 `demo.html`/`demo.png` — 문서 오류 발견)
- [x] quantization format 2종 이상 비교하여 크기·정확도 트레이드오프를 수치로 설명 가능 (smoke-test 3종 + arr-2-eff-sm FP32/INT8, §1 레이어 C)
- [x] FLOPS/파라미터 수치를 zoo 문서와 대조하여 최소 1개 모델 검증 (arr-2-eff-sm: 파라미터 18.5K≈18K, FLOPS 1.23M≈1.2M 일치)
- [x] `model_buffer.h` TFLM 헤더 구조 설명 가능 (TFL3 매직/길이 상수/정렬은 펌웨어 책임)
- [ ] Apollo510B에 heartkit 데모를 빌드/플래싱하고 `evb` 백엔드(또는 Tileio 앱 경로)로 실물 데모 실행 성공 — **보류 중** (§2~3 참고: `AmbiqAI/tileio-demos`의 `apollo510_evb` 플랫폼용 앱 이식이 남은 과제, neuralSPOT SDK 자체 지원은 확인됨)
- [ ] (선택) beat/denoise 등 다른 태스크 스모크 테스트 1회 이상 — 미착수

---

## 5. 태스크 & 모드 상세 (레이어 A 심화)

§1 레이어 A에서 확인한 구조를 실제 라벨/코드 수준까지 파고든 내용. (2026-09-02 추가)

### 5-1. 7개 태스크 — 각각 무엇을 하는가

같은 심전도(ECG) 신호를 넣어도 "무엇을 알아내고 싶은가"에 따라 태스크가 갈린다. 각 태스크의 라벨 정의(`heartkit/tasks/<task>/defines.py`)를 직접 확인한 내용:

| 태스크 | 하는 일 | 입력 → 출력 (실제 클래스, `defines.py` 기준) |
| --- | --- | --- |
| **Denoise** | 신호에서 잡음 제거 | 잡음 낀 ECG → 깨끗한 ECG (분류가 아니라 신호 자체를 복원) |
| **Segmentation** | 심장 박동 한 주기를 구간별로 나눔 | `HKSegment`: `normal, pwave, qrs, twave, uwave(미사용), noise, systolic, diastolic` |
| **Rhythm** | 심장 박동 리듬(부정맥) 판별 | `HKRhythm`: `sr(동성리듬), sbrad, stach, sarrh, svarr, svt, vtach, afib, aflut, vfib` — `arr-2-eff-sm`은 이 중 정상 vs AFIB/AFL 2종만 구분하는 축소판 |
| **Beat** | 심장 박동 한 박 한 박이 정상인지 판별 | `HKBeat`: `normal, pac(조기심방수축), pvc(조기심실수축), noise` |
| **Diagnostic** | 심전도로 질병 카테고리 진단 (multi-label 가능) | `HKDiagnostic`: `NORM, STTC, MI, HYP, CD` |
| **Foundation** | 다른 태스크들의 사전학습(pretraining) 기반 모델 생성 | 라벨 없음. 같은 신호를 두 가지로 증강해서 "같은 원본에서 왔음"을 맞추도록 학습(SimCLR 방식) — 다른 태스크가 이 모델을 가져다 미세조정 |
| **Translate** | 한 종류 신호를 다른 종류로 변환 | ECG ↔ PPG 상호 변환 |

### 5-2. 태스크는 어떻게 만들어지는가 (레시피)

태스크 하나는 `heartkit/tasks/<태스크이름>/` 폴더 안 4가지 재료로 구성된다:

1. **`defines.py`** — 위 표의 클래스 목록을 `IntEnum`으로 정의 (예: `HKRhythm.afib = 7`)
2. **데이터셋 연결부(`datasets.py`, `dataloaders/`)** — 어떤 원본 데이터셋(ptbxl, lsad, ludb 등)에서 이 태스크에 맞는 조각을 어떻게 잘라올지
3. **모델 아키텍처 선택** — 코드가 아니라 **config json**에서 지정 (`configs/arr-2-eff-sm.json`의 `"architecture": {"name": "efficientnetv2", ...}`)
4. **`train.py`/`evaluate.py`/`export.py`/`demo.py`** — 실제 파이프라인 함수. `<Task>Task(HKTask)` 클래스가 이 함수들을 그대로 위임 호출하는 얇은 어댑터

이 4가지를 `HKTask` 틀에 꽂으면 태스크 완성. 8번째 태스크를 새로 만들고 싶으면 `HKTask`를 상속해 4개 메서드(train/evaluate/export/demo)만 구현하면 됨 — 공식 문서의 **BYOT(Bring-Your-Own-Task)** 가 이 방법.

### 5-3. 5개 모드 — 실제로 무슨 일이 벌어지나

같은 태스크 안에서 "지금 뭘 하고 싶은가"에 따라 고르는 5단계. **7개 태스크 전부에 공통된 틀**이고, 태스크마다 내용물(데이터·라벨)만 다르다 — 그래서 `-t` 뒤의 태스크 이름만 바꾸면 명령어 형태는 동일.

| 모드 | 비유 | 실제로 코드가 하는 일 |
| --- | --- | --- |
| **download** | 장보기 | config에 적힌 데이터셋들을 하나씩 `.download()`. 태스크 공통 로직(`HKTask.download()`, `heartkit/tasks/task.py`)이 처리 — 태스크별 override 불필요 |
| **train** | 요리하기 | 데이터셋을 train/val로 분할 → config의 `architecture.name`으로 모델을 새로 만들거나 기존 모델 로드 → 옵티마이저·손실함수 설정 후 `model.fit()` → `model.keras`로 저장 |
| **evaluate** | 맛보기(품질 검사) | 테스트 데이터로 저장된 모델을 채점(ACC/F1/LOSS). 학습 없이 zoo 모델만 채점할 때 쓴 것이 이 모드 (§0 참고) |
| **export** | 포장/소분하기 | 학습된 모델 → `model.tflite`(TFLite) + `model_buffer.h`(TFLM C 헤더) 동시 변환. 변환 전후 정확도(TF METRICS vs TFL METRICS) 자동 비교 (§1 레이어 C의 양자화 비교가 이 단계) |
| **demo** | 시식회 | 신호 하나를 흘려보내며 `pc`/`evb` 중 고른 백엔드로 추론 실행 → 그래프/리포트 생성 (§1 레이어 B에서 확인한 `demo.html`) |

### 5-4. Segmentation은 "비트 단위"가 아니라 "샘플 단위" 라벨링

**질문**: 세그멘테이션이 연속 시계열을 한 주기(비트)로 나누는 거라면, 라벨도 비트별로 붙는가?

**답: 아니다.** `heartkit/tasks/segmentation/dataloaders/ludb.py`에서 라벨이 만들어지는 실제 코드:

```python
labels = np.zeros_like(data)                    # 원본 신호와 "똑같은 길이"의 배열
labels[seg_start:seg_end, lead] = seg_label      # 구간의 시작~끝 "샘플 전부"에 같은 라벨을 칠함
```

즉 신호가 1000개 샘플이면 라벨도 1000개짜리 배열이고, **매 샘플마다** `normal/P파/QRS/T파` 중 하나가 찍힌다. 이미지 세그멘테이션에서 픽셀 하나하나에 클래스를 칠하는 것과 동일한 방식 — 여기서는 "픽셀"이 "시간축의 샘플 하나"다.

```
원본 신호:      ─╱╲──╱╲╲╱────╱╲──╱╲╲╱───
샘플별 라벨:     000011122000000011122000   (0=무의미, 1=P파, 2=QRS, 3=T파)
```

학습 조각 하나(`frame_size`, 예: 256샘플)에 대해 모델은 256개 샘플 각각에 클래스를 하나씩 예측한다 (`demo.py`의 `class_shape = (frame_size, num_classes)`가 이를 보여줌).

**Beat 태스크와 헷갈리기 쉬운 지점**: "비트 하나에 라벨 하나"는 오히려 **Beat 태스크** 방식이다.

| | Segmentation | Beat |
| --- | --- | --- |
| 라벨 단위 | 신호의 **매 샘플** | **박동 하나 전체**에 라벨 1개 |
| 비유 | 사진 픽셀마다 색칠 | 사진 1장에 태그 1개 |
| 예시 | "3200~3250번째 샘플 = QRS" | "이번 박동 = PVC" |

세그멘테이션 결과(샘플별 마스크)에서 "QRS로 칠해진 구간의 중심"을 찾으면 그게 R-피크(박동 위치)가 된다 — `demo.py`의 R-peak 추출 로직이 정확히 이 QRS 구간 중심 탐색이었다. 즉 세그멘테이션 자체는 비트 단위가 아니지만, **세그멘테이션 결과로부터 비트 위치를 역산**하는 데 쓰인다 (§1 레이어 B에서 본 HRV/R-peak 리포트가 이 경로로 만들어짐).
