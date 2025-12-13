import os
import pandas as pd
from collections import defaultdict

# 데이터셋을 불러오는 코드 (csv 전용)
def load_dataset(input_path):
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"지원하지 않는 파일 확장자입니다: {ext}")

    return df


# 유효한 데이터만 통과시키는 함수 (유효 여부 + 탈락 사유 반환)
def is_valid_row(row):
    # 1) mbti 체크 (f/t가 아닌것 = 탈락)
    mbti = str(row.get("mbti", ""))
    if mbti not in {"f", "t"}:
        return False, "mbti_invalid"

    # 2) conversation / response 존재 & 빈 문자열 아님
    conversation = row.get("conversation", "")
    response = row.get("response", "")

    if pd.isna(conversation):
        return False, "conversation_nan"  # 대화 데이터 누락
    if pd.isna(response):
        return False, "response_nan"  # 응답 데이터 누락

    conversation = str(conversation).strip()
    response = str(response).strip()

    if conversation == "":
        return False, "conversation_empty"  # 대화 데이터 빈 문자열
    if response == "":
        return False, "response_empty"  # 응답 데이터 빈 문자열

    # 3) 길이 체크: 5글자 이하 필터링
    if len(conversation) <= 5:
        return False, "conversation_too_short"
    if len(response) <= 5:
        return False, "response_too_short"

    return True, "valid"


# Dataframe에서 유효한 행만 남기는 함수 (+ 탈락 사유별 통계 출력)
def filter_dataset(df):
    before = len(df)

    stats = defaultdict(int)
    valid_rows = []

    for _, row in df.iterrows():
        ok, reason = is_valid_row(row)
        stats[reason] += 1
        if ok:
            valid_rows.append(row)

    df_valid = pd.DataFrame(valid_rows).copy()
    after = len(df_valid)

    print("\n=== 데이터 필터링 결과 ===")
    print(f"전체 데이터: {before}개")
    print(f"유효 데이터: {after}개")
    print(f"제외된 데이터: {before - after}개")

    print("\n[탈락 사유별 개수]")
    for reason, count in sorted(stats.items(), key=lambda x: (-x[1], x[0])):
        if reason != "valid":
            print(f"- {reason}: {count}개")

    return df_valid


# 데이터셋을 train/test셋으로 나누는 함수
def split_train_test(df, train_ratio=0.8):
    df_shuffled = df.sample(frac=1.0).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)

    df_train = df_shuffled.iloc[:n_train].reset_index(drop=True)
    df_test = df_shuffled.iloc[n_train:].reset_index(drop=True)

    print(f"\n학습 데이터: {len(df_train)}개, 테스트 데이터: {len(df_test)}개")

    return df_train, df_test


# csv 데이터 저장 함수
def save_as_csv(df_train, df_test):
    # 한글 깨짐 방지를 위해 utf-8-sig 사용
    df_train.to_csv("train.csv", index=False, encoding="utf-8-sig")
    df_test.to_csv("test.csv", index=False, encoding="utf-8-sig")

    print("\n저장 완료: train.csv")
    print("저장 완료: test.csv")


def main():
    df = load_dataset("mbti_train.csv")

    # 컬럼 목록 출력
    print("컬럼 목록:", list(df.columns))

    # 조건에 맞는 데이터만 필터링 (+ 탈락 사유별 통계 출력)
    df_valid = filter_dataset(df)

    # train / test split
    df_train, df_test = split_train_test(df_valid, train_ratio=0.8)

    # CSV로 저장
    save_as_csv(df_train, df_test)


if __name__ == "__main__":
    main()
