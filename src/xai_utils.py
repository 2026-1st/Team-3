from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from models.model_config import MODEL_FEATURES, TARGET_COL


CLASS_NAMES = ["1단계_완전소외", "2단계_부분적응", "3단계_자립적응"]
RANDOM_STATE = 42


_AI_LABEL_MAP = {
    "AI_인지": "AI 인지",
    "AI_사용빈도": "AI 사용 빈도",
    "AI_도움정도": "AI 도움 정도",
}

_SECTION_PREFIX_MAP = {
    "PC 이용 능력": "PC",
    "모바일기기 이용 능력": "모바일",
    "디지털 역량 진단지수 중 활용역량": "",
    "디지털역량 진단지수 중 참여/예방/소양역량": "",
    "검색 및 이메일, 콘텐츠 서비스 이용 비율_PC": "",
    "사회적 관계 서비스 이용 비율_PC": "",
    "생활 서비스 이용 비율_PC": "",
    "정보생산 공유 비율_PC": "",
    "네트워킹 비율_PC": "",
    "사회참여 비율_PC": "",
    "경제활동 비율_PC": "",
    "디지털 기술에 대한 태도": "",
    "디지털 기기 이용 효능감": "",
}

_EXACT_BODY_MAP = {
    "필요한 프로그램을 컴퓨터에 설치/삭제/업데이트 할 수 있다": "프로그램 설치·업데이트",
    "PC에 유선 또는 무선 인터넷을 스스로 연결해서 사용할 수 있다": "인터넷 연결",
    "웹 브라우저에서 내가 원하는 환경을 설정할 수 있다": "브라우저 설정",
    "PC에 다양한 외장기기를 연결하여 이용할 수 있다": "외장기기 연결",
    "PC에 있는 파일을 인터넷을 통해 다른 사람에게 전송할 수 있다": "파일 전송",
    "PC의 악성코드를 검사/치료할 수 있다": "악성코드 검사",
    "스마트기기에서 디스플레이/소리/보안/알림/입력방법 등의 환경설정을 할 수 있다": "모바일 환경설정",
    "스마트기기에서 무선 랜 설정을 할 수 있다": "와이파이 설정",
    "스마트기기에 있는 파일을 PC로 옮길 수 있다": "모바일 파일 PC 이동",
    "스마트기기에 있는 파일/사진 등을 다른 사람에게 전송할 수 있다": "모바일 파일 전송",
    "앱을 스마트기기에 설치/삭제/업데이트할 수 있다": "앱 설치·업데이트",
    "스마트기기의 악성코드를 검사/치료할 수 있다": "모바일 악성코드 검사",
    "스마트기기 도구용 앱 이용": "도구 앱 이용",
    "비대면 원격회의앱을 통해 회의 개최/참여": "원격회의 참여",
    "스마트워치, 스마트냉장고, 사물인터넷 기기 이용": "IoT 기기 이용",
    "디지털콘텐츠 파일 편집 또는 형식 변경": "콘텐츠 편집",
    "과제나 업무를 위한 협업": "업무·과제 협업",
    "온라인 간편결제를 이용한 물건 구매": "간편결제 구매",
    "커뮤니티를 찾아 참여": "커뮤니티 참여",
    "정치/사회 문제 토론 또는 온라인기반 사회/행정서비스로 의견 개진": "온라인 의견 개진",
    "PC 및 스마트폰 보안설정": "보안 설정",
    "쿠키 및 방문기록 삭제": "기록 삭제",
    "개인정보 및 게시글에 대한 공개범위 파악/설정": "공개범위 설정",
    "권리 침해 시 조치 또는 신고 방법 인지": "권리침해 대응 인지",
    "최근 인터넷 이용 시기": "최근 인터넷 이용",
    "정보 및 뉴스 검색": "정보·뉴스 검색",
    "이메일": "이메일",
    "미디어콘텐츠 이용": "미디어 콘텐츠",
    "교육콘텐츠": "교육 콘텐츠",
    "SNS": "SNS",
    "메신저": "메신저",
    "개인 블로그": "블로그",
    "커뮤니티": "커뮤니티",
    "클라우드 서비스": "클라우드",
    "생활정보서비스": "생활정보",
    "전자상거래서비스": "전자상거래",
    "금융거래서비스": "금융거래",
    "공공서비스": "공공서비스",
    "디지털헬스케어서비스": "디지털 헬스케어",
    "직접 만들거나 다른 사람이 만든 것을 수정/편집한 콘텐츠": "콘텐츠 제작·편집",
    "인터넷에서 본 콘텐츠를 올리거나 링크를 공유한 적이 있다": "콘텐츠 공유",
    "기존에 알던 사람들과 관계를 유지하고 더 친밀해지기 위해서 인터넷을 이용한 적이 있다": "기존 관계 유지",
    "새로운 사람들을 알게 되고 소통하기 위해 인터넷을 이용한 적이 있다": "새로운 사람과 소통",
    "인터넷을 통해 사회적 관심사에 대해 의견 표명을 한 적이 있다": "사회 이슈 의견표명",
    "인터넷을 통해 정부/지자체/공공기관에 정책제안이나 건의, 정책평가, 민원제기 등을 한 적이 있다": "정책 제안·민원",
    "인터넷을 통해 기부나 봉사 활동을 한 적이 있다": "기부·봉사 참여",
    "인터넷을 통해 온라인 투표나 여론조사, 서명 등에 참여한 적이 있다": "온라인 투표·서명",
    "인터넷을 통해 취업이나 이직에 도움이 되는 활동을 한 적이 있다": "취업·이직 활동",
    "인터넷을 통해 창업이나 사업에 도움이 되는 마케팅 활동을 한 적이 있다": "사업·마케팅 활동",
    "인터넷을 통해 소득증대에 도움이 되는 관련 정보검색/습득, 재테크 등의 활동을 한 적이 있다": "소득·재테크 활동",
    "인터넷을 통해 비용절감에 도움이 되는 활동을 한 적이 있다": "비용절감 활동",
    "디지털 기술은 유용하다": "디지털 기술 유용성",
    "디지털 기술은 내 삶을 편리하게 한다": "디지털 기술 편리성",
    "디지털 기술은 나에게 좋은 것이다": "디지털 기술 긍정 인식",
    "디지털 기술을 더 많이 이용하고 싶다": "디지털 기술 이용 의향",
    "나는 디지털 기기를 배우는데 자신이 있다": "기기 학습 자신감",
    "나는 디지털 기기를 활용하는데 자신이 있다": "기기 활용 자신감",
    "나는 새로운 디지털 기기의 사용방법을 빠르게 알아낼 수 있다": "새 기기 적응 자신감",
    "디지털 기기를 더 많이 이용하고 싶다": "기기 이용 의향",
}

_BODY_REGEX_REPLACEMENTS = [
    (r"\([^)]*\)", ""),
    (r"크롬, 사파리, 엣지, 웨일 등", ""),
    (r"와이파이, 기가와이파이 포함", ""),
    (r"바이러스, 스파이웨어 등", ""),
    (r"정보/지식/뉴스/동영상/사진 등", ""),
    (r"공동구매, 해외직접구매, 가격비교 등", ""),
    (r"금전/재능", ""),
    (r"댓글 작성, 게시판 글 게시, 토론 등", ""),
    (r"홍보, 광고, 판촉, 프로모션 등", ""),
]

_BODY_PATTERN_MAP = [
    (r"필요한 프로그램.*설치/삭제/업데이트", "프로그램 설치·업데이트"),
    (r"유선 또는 무선 인터넷.*연결", "인터넷 연결"),
    (r"웹 브라우저.*환경을 설정", "브라우저 설정"),
    (r"외장기기.*연결", "외장기기 연결"),
    (r"PC에 있는 파일.*다른 사람에게 전송", "파일 전송"),
    (r"악성코드.*검사/치료", "악성코드 검사"),
    (r"스마트기기.*환경설정", "모바일 환경설정"),
    (r"무선 랜.*설정", "와이파이 설정"),
    (r"(스마트기기|모바일)에 있는 파일.*PC로 옮", "모바일 파일 PC 이동"),
    (r"(스마트기기|모바일)에 있는 파일/사진.*전송", "모바일 파일 전송"),
    (r"앱을 (스마트기기|모바일)에 설치/삭제/업데이트", "앱 설치·업데이트"),
    (r"스마트워치.*사물인터넷 기기 이용", "IoT 기기 이용"),
    (r"디지털콘텐츠 파일 편집.*형식 변경", "콘텐츠 편집"),
    (r"과제나 업무를 위한 협업", "업무·과제 협업"),
    (r"온라인 간편결제.*물건 구매", "간편결제 구매"),
    (r"정치/사회 문제 토론.*의견 개진", "온라인 의견 개진"),
    (r"커뮤니티를 찾아 참여", "커뮤니티 참여"),
    (r"정보 및 뉴스 검색", "정보·뉴스 검색"),
    (r"미디어콘텐츠", "미디어 콘텐츠"),
    (r"교육콘텐츠", "교육 콘텐츠"),
    (r"직접 만들거나 다른 사람이 만든 것을 수정/편집한 콘텐츠", "콘텐츠 제작·편집"),
    (r"인터넷에서 본 콘텐츠.*올리거나 링크를 공유", "콘텐츠 공유"),
    (r"기존에 알던 사람들과 관계를 유지.*인터넷을 이용", "기존 관계 유지"),
    (r"새로운 사람들을 알게 되고 소통하기 위해.*인터넷을 이용", "새로운 사람과 소통"),
    (r"사회적 관심사.*의견 표명", "사회 이슈 의견표명"),
    (r"정부/지자체/공공기관.*정책제안.*민원제기", "정책 제안·민원"),
    (r"기부.*봉사 활동", "기부·봉사 참여"),
    (r"온라인 투표.*여론조사.*서명", "온라인 투표·서명"),
    (r"취업이나 이직.*도움이 되는 활동", "취업·이직 활동"),
    (r"창업이나 사업.*마케팅 활동", "사업·마케팅 활동"),
    (r"소득증대.*재테크.*활동", "소득·재테크 활동"),
    (r"비용절감.*활동", "비용절감 활동"),
]


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" _-")


def _clean_body_text(body: str) -> str:
    text = str(body).strip()
    for pattern, repl in _BODY_REGEX_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    text = _normalize_spaces(text)
    text = text.replace("스마트기기", "모바일")
    text = text.replace("모바일기기", "모바일")
    text = text.replace("디지털콘텐츠", "디지털 콘텐츠")
    text = text.replace("디지털헬스케어", "디지털 헬스케어")
    text = text.replace("온라인기반", "온라인")

    for pattern, simplified in _BODY_PATTERN_MAP:
        if re.search(pattern, text):
            return simplified

    for original, simplified in _EXACT_BODY_MAP.items():
        if text == original:
            return simplified

    text = re.sub(r"^나는\s+", "", text)
    text = re.sub(r"^내가\s+", "", text)
    text = re.sub(r"^인터넷을 통해\s+", "", text)
    text = re.sub(r"^인터넷에서 본\s+", "", text)
    text = re.sub(r"^스마트기기에서\s+", "", text)
    text = re.sub(r"^스마트기기에\s+", "", text)
    text = re.sub(r"^PC에\s+", "", text)
    text = re.sub(r"^PC의\s+", "", text)

    text = re.sub(r"\s*할 수 있다$", "", text)
    text = re.sub(r"\s*사용할 수 있다$", "", text)
    text = re.sub(r"\s*이용할 수 있다$", "", text)
    text = re.sub(r"\s*이용한 적이 있다$", "", text)
    text = re.sub(r"\s*한 적이 있다$", "", text)
    text = re.sub(r"\s*등을 한 적이 있다$", "", text)
    text = re.sub(r"\s*등을 할 수 있다$", "", text)

    text = _normalize_spaces(text)

    return text


def _simplify_question_label(feature: str, label: str) -> str:
    if feature in _AI_LABEL_MAP:
        return _AI_LABEL_MAP[feature]

    text = _normalize_spaces(str(label))
    if not text:
        return feature

    text = re.sub(r"\([^)]*합산\)", "", text).strip()
    text = re.sub(r"^문\d+\.\s*", "", text).strip()

    if ")" in text:
        prefix, body = text.split(")", 1)
        section = re.sub(r"[-_]\s*\d+\s*$", "", prefix).strip()
        section = re.sub(r"\s+\d+\s*$", "", section).strip()
        body = _clean_body_text(body)
        section_prefix = _SECTION_PREFIX_MAP.get(section, "")

        if body:
            if section_prefix:
                if body.startswith(section_prefix):
                    return body
                return _normalize_spaces(f"{section_prefix} {body}")
            return body

        if section_prefix:
            return section_prefix
        return section

    return _clean_body_text(text)


def concise_feature_label(feature: str, text: str | float | None, max_len: int = 10) -> str:
    """Return a short display label while preserving readable Korean text when possible."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return str(feature)

    label = _simplify_question_label(feature, str(text).strip())
    if not label:
        return str(feature)

    label = _normalize_spaces(label)
    if len(label) <= max(max_len, 14):
        return label

    words = [word for word in label.split(" ") if word]
    if len(words) >= 2:
        compact = " ".join(words[:2])
        if len(compact) <= max(max_len, 14):
            return compact

    if feature.startswith("AI_") and feature in _AI_LABEL_MAP:
        ai_label = _AI_LABEL_MAP[feature]
        if len(ai_label) <= max(max_len, 14):
            return ai_label

    return label[: max(max_len, 14)].rstrip()


def make_unique_display_labels(
    features: Sequence[str],
    labels: Sequence[str],
    base_labels: Sequence[str],
) -> list[str]:
    """Attach the original feature code to every display label for consistent plotting."""
    del labels  # The notebook passes labels, but uniqueness is driven by the short labels.

    unique_labels: list[str] = []

    for feature, base_label in zip(features, base_labels):
        base = str(base_label).strip() or str(feature)
        if str(feature).startswith("AI_"):
            unique_labels.append(base)
            continue
        suffix = feature.replace("_", "") if str(feature).startswith("Q") else str(feature)
        unique_labels.append(f"{base} ({suffix})")

    return unique_labels


def _load_feature_metadata(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    meta_path = root / "models" / "MLP" / "models_feature_selection.csv"

    if not meta_path.exists():
        return pd.DataFrame({"feature": MODEL_FEATURES, "label": MODEL_FEATURES})

    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    meta = meta[["feature", "label"]].drop_duplicates("feature").copy()
    meta["base_short"] = meta.apply(
        lambda row: concise_feature_label(row["feature"], row["label"], max_len=10),
        axis=1,
    )
    meta["label_short"] = make_unique_display_labels(
        meta["feature"].tolist(),
        meta["label"].tolist(),
        meta["base_short"].tolist(),
    )
    return meta


def _feature_display_names(root: str | Path) -> list[str]:
    meta = _load_feature_metadata(root)
    label_map = dict(zip(meta["feature"], meta["label_short"]))
    return [label_map.get(feature, feature) for feature in MODEL_FEATURES]


def _load_best_c(root: str | Path, default: float = 100.0) -> float:
    summary_path = Path(root) / "models" / "SVM" / "svm_summary.json"
    if not summary_path.exists():
        return default

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary.get("best_C", default))


def _make_linear_svc(c_value: float) -> LinearSVC:
    return LinearSVC(C=c_value, dual=False, max_iter=5000, random_state=RANDOM_STATE)


def _make_calibrated_svc(c_value: float):
    base = _make_linear_svc(c_value)
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def train_final_svm_for_xai(root: str | Path) -> dict[str, object]:
    """Train the final SVM used for XAI analysis and return all intermediate artifacts."""
    from sklearn.preprocessing import StandardScaler

    root = Path(root)
    data_dir = root / "data" / "labeled"

    train_df = pd.read_csv(data_dir / "train_labeled.csv")
    val_df = pd.read_csv(data_dir / "val_labeled.csv")
    test_df = pd.read_csv(data_dir / "test_labeled.csv")

    train_full = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    X_train = train_full[MODEL_FEATURES].to_numpy(dtype="float32")
    X_test = test_df[MODEL_FEATURES].to_numpy(dtype="float32")

    y_train_raw = train_full[TARGET_COL].to_numpy()
    y_test_raw = test_df[TARGET_COL].to_numpy()
    stage_min = int(min(y_train_raw.min(), y_test_raw.min()))
    y_train = (y_train_raw - stage_min).astype(int)
    y_test = (y_test_raw - stage_min).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")

    best_c = _load_best_c(root)

    final_svc = _make_linear_svc(best_c)
    final_svc.fit(X_train_scaled, y_train)

    calibrated_svc = _make_calibrated_svc(best_c)
    calibrated_svc.fit(X_train_scaled, y_train)

    test_preds = final_svc.predict(X_test_scaled)
    test_probs = calibrated_svc.predict_proba(X_test_scaled)

    return {
        "root": root,
        "feature_names": MODEL_FEATURES,
        "feature_display_names": _feature_display_names(root),
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "final_svc": final_svc,
        "calibrated_svc": calibrated_svc,
        "test_preds": test_preds,
        "test_probs": test_probs,
    }


def create_lime_explainer(
    x_train_raw: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str] = CLASS_NAMES,
):
    """Create a LIME tabular explainer lazily so importing this module never requires lime."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME is not installed. Run `pip install lime` in the notebook environment first."
        ) from exc

    return LimeTabularExplainer(
        np.asarray(x_train_raw, dtype="float32"),
        feature_names=list(feature_names),
        class_names=list(class_names),
        discretize_continuous=True,
        mode="classification",
        random_state=RANDOM_STATE,
    )


def select_representative_case_indices(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    pred_proba: np.ndarray,
    target_classes: Iterable[int] = (0, 1, 2),
) -> dict[int, int]:
    """Pick confident correctly-classified examples per target class."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    pred_proba_arr = np.asarray(pred_proba)

    selected: dict[int, int] = {}
    for class_idx in target_classes:
        correct_mask = (y_true_arr == class_idx) & (y_pred_arr == class_idx)
        candidate_mask = correct_mask
        if not candidate_mask.any():
            candidate_mask = y_pred_arr == class_idx
        if not candidate_mask.any():
            candidate_mask = np.ones(len(y_pred_arr), dtype=bool)

        candidate_indices = np.flatnonzero(candidate_mask)
        best_local = np.argmax(pred_proba_arr[candidate_indices, class_idx])
        selected[int(class_idx)] = int(candidate_indices[best_local])

    return selected


def explain_with_lime(
    lime_explainer,
    predict_fn,
    sample_raw: Sequence[float],
    predicted_label: int,
    num_features: int = 10,
) -> pd.DataFrame:
    def _humanize_lime_rule(rule_text: str) -> str:
        text = _normalize_spaces(rule_text)
        ranged = re.match(r"^(-?\d+(?:\.\d+)?)\s*<\s*(.+?)\s*<=\s*(-?\d+(?:\.\d+)?)$", text)
        if ranged:
            return f"{ranged.group(2)} 보통"

        upper = re.match(r"^(.+?)\s*<=\s*(-?\d+(?:\.\d+)?)$", text)
        if upper:
            return f"{upper.group(1)} 낮음"

        lower = re.match(r"^(.+?)\s*[>]=?\s*(-?\d+(?:\.\d+)?)$", text)
        if lower:
            return f"{lower.group(1)} 높음"

        return text

    explanation = lime_explainer.explain_instance(
        np.asarray(sample_raw, dtype="float32"),
        predict_fn,
        labels=[predicted_label],
        num_features=num_features,
    )

    rows = []
    for rule, weight in explanation.as_list(label=predicted_label):
        display_rule = _humanize_lime_rule(rule)
        rows.append(
            {
                "rule_raw": rule,
                "rule": display_rule,
                "weight": float(weight),
                "abs_weight": abs(float(weight)),
                "direction": "positive" if weight >= 0 else "negative",
            }
        )

    return pd.DataFrame(rows).sort_values("abs_weight", ascending=False).reset_index(drop=True)


def plot_lime_explanation(
    case_df: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5.5),
) -> pd.DataFrame:
    plot_df = case_df.sort_values("weight")
    colors = plot_df["weight"].map(lambda value: "#2E8B57" if value >= 0 else "#C44E52")

    plt.figure(figsize=figsize)
    plt.barh(plot_df["rule"], plot_df["weight"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("LIME weight")
    plt.ylabel("Feature rule")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()

    return case_df


def plot_lime_case_comparison(
    case_frames: Sequence[pd.DataFrame],
    case_titles: Sequence[str],
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    fig, axes = plt.subplots(
        nrows=len(case_frames),
        ncols=1,
        figsize=(10, max(4.5, 4.2 * len(case_frames))),
        constrained_layout=True,
    )
    if len(case_frames) == 1:
        axes = [axes]

    combined_frames = []
    for ax, case_df, case_title in zip(axes, case_frames, case_titles):
        plot_df = case_df.sort_values("weight")
        colors = plot_df["weight"].map(lambda value: "#2E8B57" if value >= 0 else "#C44E52")
        ax.barh(plot_df["rule"], plot_df["weight"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(case_title, fontsize=14, fontweight="bold")
        ax.set_xlabel("LIME weight")
        ax.set_ylabel("Feature rule")

        tagged = case_df.copy()
        tagged["case_title"] = case_title
        combined_frames.append(tagged)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()

    return pd.concat(combined_frames, ignore_index=True)


def create_shap_linear_explainer(
    linear_model,
    x_background_scaled: np.ndarray,
    max_background: int = 1000,
):
    """Create a SHAP explainer lazily so the module still imports without shap installed."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Run `pip install shap` in the notebook environment first."
        ) from exc

    background = np.asarray(x_background_scaled, dtype="float32")
    if len(background) > max_background:
        background = background[:max_background]

    return shap.LinearExplainer(linear_model, background)


def _normalize_shap_values(raw_values, class_count: int) -> np.ndarray:
    values = raw_values.values if hasattr(raw_values, "values") else raw_values

    if isinstance(values, list):
        stacked = np.stack(values, axis=0)  # (class, sample, feature)
        return np.transpose(stacked, (1, 0, 2))

    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")

    if arr.shape[1] == class_count:
        return arr
    if arr.shape[2] == class_count:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[0] == class_count:
        return np.transpose(arr, (1, 0, 2))

    return arr


def _normalize_base_values(raw_values, sample_count: int, class_count: int) -> np.ndarray | None:
    base_values = getattr(raw_values, "base_values", None)
    if base_values is None:
        return None

    base = np.asarray(base_values)
    if base.ndim == 0:
        return np.full((sample_count, class_count), float(base))
    if base.ndim == 1:
        if len(base) == class_count:
            return np.tile(base, (sample_count, 1))
        if len(base) == sample_count:
            return base[:, np.newaxis]
    if base.ndim == 2:
        if base.shape == (sample_count, class_count):
            return base
        if base.shape == (class_count, sample_count):
            return base.T
    return None


def compute_shap_bundle(
    shap_explainer,
    x_scaled: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    max_samples: int | None = None,
) -> dict[str, object]:
    x_arr = np.asarray(x_scaled, dtype="float32")
    if max_samples is not None and len(x_arr) > max_samples:
        x_arr = x_arr[:max_samples]

    raw_values = shap_explainer(x_arr)
    values = _normalize_shap_values(raw_values, class_count=len(class_names))
    base_values = _normalize_base_values(raw_values, sample_count=len(x_arr), class_count=values.shape[1])

    return {
        "values": values,
        "base_values": base_values,
        "samples": x_arr,
        "class_names": list(class_names),
    }


def plot_shap_global_importance(
    shap_bundle: dict[str, object],
    feature_names: Sequence[str],
    save_path: str | Path | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    values = np.asarray(shap_bundle["values"])
    mean_abs_shap = np.abs(values).mean(axis=(0, 1))

    global_df = (
        pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plot_df = global_df.head(top_n).sort_values("mean_abs_shap")

    plt.figure(figsize=(9, 5.8))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4C78A8")
    plt.title("SVM SHAP 전역 중요도", fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()

    return global_df


def plot_shap_class_heatmap(
    shap_bundle: dict[str, object],
    feature_names: Sequence[str],
    save_path: str | Path | None = None,
    top_n: int = 12,
) -> pd.DataFrame:
    values = np.asarray(shap_bundle["values"])
    class_names = list(shap_bundle["class_names"])

    per_class_mean = np.abs(values).mean(axis=0)  # (class, feature)
    global_mean = per_class_mean.mean(axis=0)
    top_indices = np.argsort(global_mean)[::-1][:top_n]

    heatmap_df = pd.DataFrame(
        per_class_mean[:, top_indices],
        index=class_names,
        columns=np.asarray(feature_names)[top_indices],
    )

    fig_width = max(1.15 * top_n + 3.5, 14)
    fig_height = max(5.8, 1.45 + 1.15 * len(class_names))
    annot_size = 11 if top_n <= 10 else 9
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        annot_kws={"size": annot_size},
    )
    plt.title("SVM SHAP 클래스별 중요도", fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("Feature")
    plt.ylabel("Class")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.tick_params(axis="y", labelsize=12, pad=10)
    ax.tick_params(axis="x", labelsize=10, pad=8)
    plt.subplots_adjust(left=0.22, bottom=0.34, right=0.96, top=0.88)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()

    return heatmap_df


def prepare_shap_case_frame(
    shap_bundle: dict[str, object],
    feature_names: Sequence[str],
    sample_index: int,
    class_index: int,
    top_n: int = 10,
) -> pd.DataFrame:
    values = np.asarray(shap_bundle["values"])
    sample_values = values[sample_index, class_index]
    sample_features = np.asarray(shap_bundle["samples"])[sample_index]

    case_df = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "feature_value": sample_features,
                "shap_value": sample_values,
                "abs_shap": np.abs(sample_values),
            }
        )
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    case_df["direction"] = np.where(case_df["shap_value"] >= 0, "positive", "negative")
    return case_df


def plot_shap_case_explanation(
    case_df: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5.5),
) -> pd.DataFrame:
    plot_df = case_df.sort_values("shap_value")
    colors = plot_df["shap_value"].map(lambda value: "#2E8B57" if value >= 0 else "#C44E52")

    plt.figure(figsize=figsize)
    plt.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()

    return case_df
