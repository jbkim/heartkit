# heartKIT 설치 문제 해결 & 동작 확인 가이드 (macOS / Apple Silicon)

> GitHub에서 클론해서 `uv sync` 까지는 됐는데 `heartkit` 명령이 안 될 때 읽는 문서.
> 2026-08-26 기준, heartKIT 1.8.0 / Python 3.12.9 / macOS(arm64) 에서 실제로 재현하고 고친 내용입니다.

---

## 0. 세 줄 요약

1. macOS에서는 `uv sync` 가 **깨진 GPU 플러그인(`tensorflow-metal`)** 을 같이 설치해서, `heartkit --help` 조차 실행되지 않습니다.
2. 플러그인을 지우면 해결됩니다: `uv pip uninstall --python .venv/bin/python tensorflow-metal`
3. 잘 고쳐졌는지는 [configs/smoke-test.json](../configs/smoke-test.json) 으로 `train → evaluate → export` 를 1분 안에 돌려서 확인합니다.

---

## 1. 설치

```bash
git clone https://github.com/AmbiqAI/heartkit.git
cd heartkit
uv sync
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\activate
```

`uv sync` 는 `.venv/` 라는 가상환경 폴더를 만들고 그 안에 모든 패키지를 넣습니다.
`source .venv/bin/activate` 를 해야 터미널이 그 환경을 바라보게 되고, 그때부터 `heartkit` 명령을 쓸 수 있습니다.

> **활성화가 귀찮다면** `heartkit` 대신 `.venv/bin/heartkit` 또는 `uv run heartkit` 을 쓰면 됩니다. 결과는 같습니다.

---

## 2. 증상 — `heartkit` 이 아예 실행되지 않음

`heartkit --help` 처럼 아무것도 안 하는 명령조차 아래처럼 실패합니다.

```
tensorflow.python.framework.errors_impl.NotFoundError:
dlopen(.../site-packages/tensorflow-plugins/libmetal_plugin.dylib, 0x0006):
  Library not loaded: @rpath/_pywrap_tensorflow_internal.so
  Referenced from: .../tensorflow-plugins/libmetal_plugin.dylib
  Reason: tried: '.../_solib_darwin_arm64/.../_pywrap_tensorflow_internal.so' (no such file)
```

**에러를 읽는 법**

| 조각 | 의미 |
| --- | --- |
| `NotFoundError` | 파이썬 코드 버그가 아니라 **파일을 못 찾은** 문제 |
| `dlopen(...libmetal_plugin.dylib)` | 못 연 대상은 macOS **GPU 가속 플러그인** |
| `Library not loaded: @rpath/_pywrap_tensorflow_internal.so` | 그 플러그인이 요구하는 TensorFlow 내부 파일이 없음 |
| `(no such file)` | 즉, 플러그인과 TensorFlow의 **버전이 서로 안 맞음** |

핵심은 **`heartkit` 자체의 문제가 아니라는 것**입니다. `heartkit` 은 시작할 때 TensorFlow를 불러오는데, TensorFlow가 자기 폴더 안의 `tensorflow-plugins/` 를 조건 없이 전부 로드합니다. 그 안에 깨진 플러그인이 하나 있으면 `import tensorflow` 단계에서 통째로 죽습니다.

---

## 3. 원인 — TensorFlow와 metal 플러그인의 버전 불일치

[pyproject.toml](../../pyproject.toml) 은 macOS일 때 GPU 가속 패키지를 자동으로 설치합니다.

```toml
"tensorflow>=2.20.0,<3.0",
"tensorflow-metal>=1.1.0,<2.0; sys_platform == 'darwin'",
```

설치된 조합을 확인해 보면:

```bash
ls .venv/lib/python3.12/site-packages | grep -iE "^tensorflow|^keras"
# tensorflow-2.20.0.dist-info
# tensorflow_metal-1.2.0.dist-info   <-- 문제
# keras-3.10.0.dist-info
```

`tensorflow-metal` 은 Apple이 배포하는 GPU(Metal) 가속 플러그인인데, 업데이트가 TensorFlow 본체를 따라오지 못했습니다. 설치 가능한 최신 버전(1.2.0)이 TF 2.20이 더 이상 제공하지 않는 내부 라이브러리를 참조하기 때문에, 버전 범위 `<2.0` 조건은 만족해도 실제로는 로드에 실패합니다.

---

## 4. 해결 — metal 플러그인 제거

```bash
uv pip uninstall --python .venv/bin/python tensorflow-metal
```

**`--python .venv/bin/python` 을 꼭 붙이세요.** 다른 프로젝트의 가상환경이 `VIRTUAL_ENV` 환경변수에 남아 있으면 `uv` 가 엉뚱한 환경을 건드립니다. 실제로 이 옵션 없이 실행했을 때 다음처럼 다른 환경을 잡았습니다.

```
Using Python 3.10.17 environment at: /Users/…/virtualenvs/backend-5VKfjQD9
warning: Skipping tensorflow-metal as it is not installed
```

제거되면 플러그인 폴더 자체가 사라집니다.

```bash
ls .venv/lib/python3.12/site-packages/tensorflow-plugins
# ls: ...: No such file or directory   <-- 정상
```

이제 실행됩니다.

```bash
$ heartkit --help
usage: heartkit [-h] [--mode [download|train|evaluate|export|demo]] [--task TEXT] [--config TEXT]
```

### GPU 가속을 못 쓰게 되는 것 아닌가?

맞습니다. 대신 Apple Silicon의 CPU로 돌아가는데, 이 문서의 예제나 소규모 학습에는 충분합니다. Apple이 TF 2.20을 지원하는 `tensorflow-metal` 을 내놓으면 다시 설치할 수 있습니다.

### ⚠️ `uv sync` 를 다시 돌리면 재발합니다

`uv sync` 는 `pyproject.toml` 을 기준으로 환경을 되돌리므로 `tensorflow-metal` 을 다시 설치합니다. 앞으로는 이렇게 쓰세요.

```bash
uv sync --no-install-package tensorflow-metal
```

---

## 5. 동작 확인 — 다운로드 없이 1분 만에

문서의 `heartkit -m evaluate ...` 예제는 **이미 학습된 모델과 내려받은 데이터셋이 있어야** 동작합니다. 맨바닥에서 바로 `evaluate` 를 실행하면 실패하는 게 정상입니다.

그래서 다운로드가 필요 없는 **합성 ECG 데이터셋(`ecg-synthetic`)** 으로 파이프라인 전체를 점검하는 설정을 [configs/smoke-test.json](../configs/smoke-test.json) 에 만들어 두었습니다. 실제 데이터셋(`ptbxl`, `lsad` 등)은 수 GB라 받는 데 오래 걸리지만, 합성 데이터는 코드가 그 자리에서 생성합니다.

### 5-1. 학습 (약 25초)

```bash
heartkit -m train -t segmentation -c ./notes/configs/smoke-test.json
```

```
Epoch 2/2
5/5 ━━━━━━━━━━ acc: 0.2805 - loss: 1.2439 - val_acc: 0.3642 - val_loss: 0.9305
DEBUG    Model saved to results/smoke-test/model.keras
INFO     [VAL SET] ACC=0.3642, F1=0.3841, LOSS=0.9305
INFO     #FINISHED MODE=train TASK=segmentation
```

### 5-2. 평가

```bash
heartkit -m evaluate -t segmentation -c ./notes/configs/smoke-test.json
```

```
INFO     [TEST SET] ACC=0.3739, F1=0.3883, LOSS=0.9522
INFO     #FINISHED MODE=evaluate TASK=segmentation
```

### 5-3. 내보내기 (TFLite / TFLM 변환)

```bash
heartkit -m export -t segmentation -c ./notes/configs/smoke-test.json
```

```
DEBUG    Saving TFLite model to results/smoke-test/model.tflite
DEBUG    Saving TFL micro model to results/smoke-test/model_buffer.h
INFO     [TF METRICS]  LOSS=1.3700 ACC=0.4007 F1=0.3978 IOU=0.2408
INFO     [TFL METRICS] LOSS=1.3702 ACC=0.4213 F1=0.4429 IOU=0.2498
INFO     #FINISHED MODE=export TASK=segmentation
```

세 명령 모두 마지막 줄에 `#FINISHED` 가 찍히면 설치가 정상입니다.

### 정확도가 37%인데 괜찮나요?

**괜찮습니다.** 이건 성능 측정이 아니라 **배관 점검**입니다. 빨리 끝나도록 일부러 아래처럼 줄여 놓았습니다.

| 항목 | smoke-test | 실제 설정(`seg-4-tcn-sm`) |
| --- | --- | --- |
| epochs | 2 | 150 |
| steps_per_epoch | 5 | 50 |
| 합성 환자 수 | 200 | 20,000 |
| 실제 데이터셋(LUDB) | 사용 안 함 | 사용 |
| 모델 블록 수 | 2 | 4 |

학습이 안 된 상태이므로 정확도는 무작위에 가깝습니다. 확인하려는 건 "데이터 생성 → 학습 → 저장 → 로드 → 평가 → 양자화 변환"이 **에러 없이 끝까지 도는가** 하나뿐입니다.

---

## 6. 결과물 읽기 — `results/smoke-test/`

`job_dir` 설정값에 따라 생기는 폴더입니다. (`results/` 는 `.gitignore` 대상이라 커밋되지 않습니다.)

| 파일 | 내용 |
| --- | --- |
| `configuration.json` | 실제로 사용된 설정 스냅샷. 나중에 재현할 때 이걸 보세요 |
| `model.keras` | 학습된 Keras 모델. `evaluate` / `export` 가 이 파일을 읽습니다 |
| `model.tflite` | TensorFlow Lite로 양자화 변환된 모델 |
| `model_buffer.h` / `smoke_model_buffer.h` | 펌웨어에 넣는 C 헤더(TFLM). 이름은 `tflm_file` 설정값 |
| `metrics.json` | 평가 지표 요약 (`acc`, `f1`, `loss`, `flops`, `parameters`) |
| `confusion_matrix_test.png` / `.html` | 혼동 행렬. HTML은 마우스를 올리면 수치가 보입니다 |
| `history.csv` / `history.png` | 에폭별 학습 곡선 |
| `train.log` / `test.log` / `export.log` | 모드별 전체 로그. 실패 원인은 여기부터 확인 |
| `model_flops.log` | 연산량(FLOPS). 임베디드 타깃에서는 중요한 수치 |

---

## 7. CLI 사용법 정리

```bash
heartkit --mode [MODE] --task [TASK] --config [CONFIG]
heartkit -m     [MODE] -t     [TASK] -c      [CONFIG]
```

| 인자 | 값 |
| --- | --- |
| `MODE` | `download`, `train`, `evaluate`, `export`, `demo` |
| `TASK` | `segmentation`, `rhythm`, `beat`, `denoise`, `diagnostic`, `foundation`, `translate` |
| `CONFIG` | JSON 파일 경로, 또는 JSON 문자열 그 자체 |

설정 파일 하나로 모든 모드를 돌립니다. 각 모드는 필요한 항목만 골라 씁니다.
(`--help` 에는 `MODE` 목록만 나오고 `TASK` 는 `TEXT` 로 표시되지만, 위 7개가 등록된 전부입니다 — [heartkit/tasks/\_\_init\_\_.py:59-65](../../heartkit/tasks/__init__.py#L59-L65))

---

## 8. 실제 데이터로 넘어가기

### 8-1. 데이터셋 내려받기

```bash
heartkit -m download -c ./configs/download-datasets.json
```

`icentia11k`, `ludb`, `qtdb`, `ptbxl`, `lsad` 를 `./datasets/` 아래에 받습니다. **수 GB에 수십 분** 걸립니다. 일부만 필요하면 `configs/download-datasets.json` 에서 필요 없는 항목을 지우세요.

### 8-2. 사전학습 모델(Model Zoo) 평가하기

학습 없이 Ambiq가 배포한 모델을 평가해 볼 수 있습니다.

```bash
mkdir -p results/arr-2-eff-sm
curl -o results/arr-2-eff-sm/model.keras \
  https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-2-eff-sm/latest/model.keras

heartkit -m evaluate -t rhythm -c ./configs/arr-2-eff-sm.json
```

> **⚠️ 함정: `val_file` 을 먼저 지워야 합니다.**
> `configs/arr-2-eff-sm.json` 에는 `"val_file": "val.tfds"` 가 들어 있습니다. `val.tfds` 는 `train` 을 돌려야 생기는 캐시인데, [evaluate.py:32](../../heartkit/tasks/rhythm/evaluate.py#L32) 는 파일 존재 여부를 확인하지 않고 그냥 `tf.data.Dataset.load()` 를 호출합니다.
> 학습 없이 zoo 모델만 평가하려면 설정에서 `val_file` 과 `test_file` 두 줄을 **삭제**하세요. 그러면 데이터셋에서 테스트셋을 새로 만듭니다.

모델별 URL은 [docs/zoo/](../../docs/zoo/) 의 각 문서 하단 표에 있습니다.

---

## 9. 자주 만나는 에러 대처표

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `zsh: command not found: heartkit` | 가상환경 미활성화 | `source .venv/bin/activate` 또는 `uv run heartkit ...` |
| `NotFoundError: dlopen(...libmetal_plugin.dylib)` | metal 플러그인 불일치 | 4장 — `uv pip uninstall --python .venv/bin/python tensorflow-metal` |
| 고쳤는데 `uv sync` 후 다시 같은 에러 | `uv sync` 가 재설치 | `uv sync --no-install-package tensorflow-metal` |
| `uv pip uninstall` 이 "not installed" 라고 함 | 다른 가상환경을 잡음 | `--python .venv/bin/python` 명시 |
| `evaluate` 에서 `val.tfds` 관련 에러 | 학습 캐시가 없음 | 설정에서 `val_file` / `test_file` 삭제 (8-2 참고) |
| `evaluate` 에서 모델 파일을 못 찾음 | `train` 을 안 돌림 | 먼저 `-m train`, 또는 zoo에서 `model.keras` 내려받기 |
| 데이터셋 경로 에러 | 다운로드 안 함 | `heartkit -m download -c ./configs/download-datasets.json` |
| `ValueError: Unknown task ...` | task 이름 오타 | 7장의 7개 이름 확인 (`segmentation` 을 `segment` 로 쓰는 실수 잦음) |

---

## 10. 이 문서가 만든 파일

| 경로 | git 추적 | 설명 |
| --- | --- | --- |
| `notes/configs/smoke-test.json` | 커밋됨 | 5장의 1분 점검용 설정. 필요 없으면 삭제해도 됩니다 |
| `results/smoke-test/` | `.gitignore` 대상 | 점검 실행 결과물 |
| `notes/docs/SETUP-TROUBLESHOOTING.ko.md` | 커밋됨 | 이 문서 |

`uv.lock` 의 변경은 이 작업과 무관하게 이전부터 있던 것입니다.
