# Git 제출 범위

이 폴더는 원본 실험 작업 사본에서 선별한 제출본이다. 팀 저장소 main의 `3a77ff40c313961e39c8fd88cff490e5a40ca3ec`에서 만든 `experiment/gpu03-e3-e4` 브랜치에 통합했다. 원본 작업 사본은 보존했으며 push·PR은 별도 승인 후 진행한다.

- 원본 167파일 패치를 무조건 적용하지 않았다. E3·E4 코드와 설정은 원본 그대로이고 문서만 제출 구조에 맞게 정리했다.
- 원본 archive 122개 파일을 검증한 뒤 93개 CSV/JSON/Markdown만 선별했다. 측정 CSV와 provenance/freeze/decision의 바이트는 원본과 같다.
- `ARTIFACT_SHA256.json`: 이 제출본에 실제 포함한 93개 파일. `SOURCE_ARTIFACT_SHA256.json`: 원본 전체 122개 해시 목록. 원본 목록의 생략 파일 부재를 측정 실패로 해석하지 않는다.
- 생략: 중복 inference/console/finalization 로그, 서버 환경·프로세스 덤프, 외부 vismatch 소스 사본, 초기 실패 진단/임시 결과.
- 원본 ZIP·이미지·가중치·venv·캐시·인증 정보는 포함하지 않는다. 데이터는 원본 ZIP의 선별 재구성 방법과 작은 manifest/해시만 제공한다.
- 공통 코드 변경은 `imatch/__init__.py`와 `imatch/geometry.py`의 지연 import뿐이다. 팀 최신 기본 브랜치의 원래 코드와 실측 snapshot이 같음을 대조한 뒤 반영했다. 회전·블록 선택·타인 실험 코드는 변경하지 않는다.
- README의 `runs/` 링크는 제출되는 `results/`로 수정했다. 측정 당시 source fingerprint 대상 Python/설정/requirements/weights 명세는 변경하지 않았다.

팀 저장소 코드와의 파일 비교·테스트·원래 evaluate·재계산 결과는 `validation/team_integration.json`에 기록한다. 측정 당시 Python·설정·requirements는 보존했으며 공통 파이프라인·평가 파일의 SHA256도 당시 source fingerprint와 일치해야 한다. 팀 일정 문서와 타인의 실험·기존 결과 CSV는 변경하지 않는다.
