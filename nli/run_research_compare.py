"""연구용: 근거·주장 쌍 예제로 세 경로를 비교하고 JSON·CSV로 저장한다.

- 입력: 기본은 data/nli_research/dev_pairs.jsonl (개발용). --examples·--candidate-set으로 다른 쌍 파일의 한 묶음을 준다.
  최종 평가 후보(data/nli_research/final_eval_candidates.jsonl, 설정 final_eval_config.json)는 사람이 확정한 뒤에만 쓴다.
- 모델 입력은 evidence·claim뿐이다. 참고 라벨·원문·메모는 기록에만 남긴다.
- 분류기는 선택한 경로에 필요한 것만 불러오고, 불러올 때 한 번 미리 추론해 측정에서 뺀다.
  LLM 워밍업 1회는 따로 기록하고 측정에서 뺀다.
- 예제마다 실행할 수 있는 경로의 순서를 돌려 가며 바꾼다. 실행하지 못하는 경로(모델 미준비)는 순서에서 빼고
  unavailable로 기록하며 summary 통계에서도 뺀다.
- 사람 평가 열은 비워 둔다. 자동 품질 점수는 만들지 않는다.

실행(저장소 루트):
  uv run python nli/run_research_compare.py check01 --limit 2
  uv run python nli/run_research_compare.py dev_qwen3_01 --paths llm_only roberta
결과: artifacts/step_by_step/klue_nli/wanwi/research/pipeline_compare/<이름>/results.json, results.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import research_pipeline as research

OUTPUT_ROOT = config.LEARNER_DIR / "research" / "pipeline_compare"
WARMUP = ("워밍업용 근거 문장입니다. 측정에 넣지 않습니다.", "워밍업용 주장입니다.")


def rotated_order(index, available=research.PATH_IDS):
    """예제 번호에 따라 실행할 수 있는 경로의 순서를 한 칸씩 돌린다.

    실행하지 못하는 경로(모델 미준비 등)는 회전에서 빼고 뒤에 붙인다. 두 경로만 있으면 예제마다 앞뒤가 바뀐다.
    """
    runnable = [path_id for path_id in research.PATH_IDS if path_id in available]
    if not runnable:
        return list(research.PATH_IDS)
    shift = index % len(runnable)
    rest = [path_id for path_id in research.PATH_IDS if path_id not in runnable]
    return runnable[shift:] + runnable[:shift] + rest


def run(name, llm=research.DEFAULT_LLM, paths=research.PATH_IDS, limit=None, examples_file=None,
        candidate_set=None, output_root=OUTPUT_ROOT, registry=None, call=None, log=print):
    out_dir = Path(output_root) / name
    if out_dir.exists():
        raise SystemExit(f"결과 폴더가 이미 있습니다: {out_dir}")
    examples_file = Path(examples_file or research.DEV_EXAMPLES)
    selected = [path_id for path_id in research.PATH_IDS if path_id in paths]
    examples = research.load_examples(examples_file, candidate_set)[:limit]
    registry = registry or research.ClassifierRegistry()

    log("실행 경로: " + ", ".join(selected))
    warmup = research.run_comparison(*WARMUP, llm, registry, selected=selected, call=call)
    log("워밍업: " + ", ".join(f"{p['path']}={p['status']}" for p in warmup["paths"]))
    for path in warmup["paths"]:
        if path["status"] == "unavailable":
            log(f"  {path['path']}: 사용 불가 ({path['unavailable_code']}) {path['unavailable_reason']}")

    available = [p for p in research.available_path_ids(registry) if p in selected]
    records = []
    for index, example in enumerate(examples):
        record = research.run_comparison(
            example["evidence"], example["claim"], llm, registry,
            order=rotated_order(index, available), selected=selected, call=call,
        )
        record["example"] = {key: example[key] for key in ("id", "reference_label", "source_text")}
        records.append(record)
        log(f"[{index + 1}/{len(examples)}] {example['id']} 순서={'→'.join(record['executed']) or '-'} "
            + ", ".join(f"{p['path']}={p['status']}" for p in record["paths"]))

    out_dir.mkdir(parents=True)
    summary = {
        "schema": research.SCHEMA,
        "examples_file": (
            examples_file.resolve().relative_to(research.ROOT).as_posix()
            if examples_file.resolve().is_relative_to(research.ROOT) else str(examples_file)
        ),
        "candidate_set": candidate_set,
        "llm": {"key": llm, **research.REGISTERED_LLMS[llm]},
        "selected_paths": selected,
        "generation_config": research.generation_config(llm),
        "policy_version": research.POLICY_VERSION,
        "classifiers": {name: registry.info(name) for name in registry.runs},
        "warmup": warmup,  # 측정에서 뺀 실행
        # 경로별 상태 개수와 시간·토큰 평균·중앙값. 실행하지 않은 경로는 excluded, 값은 null(해당 없음)
        "summary": research.summarize(records),
        "records": records,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다. csv 모듈이 줄 끝을 정하므로 newline=""로 쓴다.
    with open(out_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as file:
        file.write(research.to_csv(records))
    log(f"저장: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="결과 폴더 이름 (이미 있으면 멈춘다)")
    parser.add_argument("--llm", default=research.DEFAULT_LLM, choices=list(research.REGISTERED_LLMS))
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 실행")
    parser.add_argument("--paths", nargs="+", choices=research.PATH_IDS, default=list(research.PATH_IDS),
                        help="실행할 경로 (기본: 세 경로 모두). 선택하지 않은 경로의 모델은 불러오지 않는다")
    parser.add_argument("--examples", default=None, help="근거·주장 쌍 JSONL (기본: 개발용 dev_pairs.jsonl)")
    parser.add_argument("--candidate-set", default=None, help="예제 파일에서 고를 후보 묶음 (최종 평가 후보 파일용)")
    args = parser.parse_args()
    run(args.name, llm=args.llm, paths=args.paths, limit=args.limit, examples_file=args.examples,
        candidate_set=args.candidate_set)


if __name__ == "__main__":
    main()
