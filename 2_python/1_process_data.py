import pandas as pd

# 데이터셋을 불러오는 코드 (csv 전용)
def load_dataset(input_path):
    df = pd.read_csv(input_path)

    return df


# 유효한 데이터만 통과시키는 함수
def is_valid_row(row):
    # 1) mbti 체크 (f/t가 아닌것 = 제거)
    mbti = str(row.get("mbti", ""))
    if mbti not in ["f", "t"]: 
        return False

    # 2,3) conversation / response 데이터 존재 & 짧은 데이터 체크
    conversation = row.get("conversation", "")
    response = row.get("response", "")

    conversation = str(conversation).strip()
    response = str(response).strip()

    # 5글자 이하 필터링
    if len(conversation) <= 5 or len(response) <= 5:
        return False

    return True


# Dataframe에서 유효한 행만 남기는 함수
def filter_dataset(df):
    before = len(df) # 필터링 이전 데이터 길이

    df_valid = df[df.apply(is_valid_row, axis=1)]
    after = len(df_valid) # 필터링 이후 데이터 길이

    print("\n=== 데이터 필터링 결과 ===")
    print(f"전체 데이터: {before}개")
    print(f"유효 데이터: {after}개")
    print(f"제외된 데이터: {before - after}개")

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
    # 데이터셋 불러오기
    df = load_dataset("mbti_train.csv")

    # 컬럼 목록 출력
    print("컬럼 목록:", list(df.columns))

    # 조건에 맞는 데이터만 필터링
    df_valid = filter_dataset(df)

    # train/test셋 나누기
    df_train, df_test = split_train_test(df_valid, train_ratio=0.8)

    # CSV로 저장
    save_as_csv(df_train, df_test)


if __name__ == "__main__":
    main()
