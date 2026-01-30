import json
from typing import List

COT_PATH = "../data/eval_data/chain-of-thought.jsonl"
MMAU_TEST_MINI_PATH = "../../MMAU/mmau-test-mini.json"


def textual_audio_to_label(textual_audio: str) -> dict:
    """
    return a dict: {
        "ASR": transcript,
        "SER": emotion,
        "GR": gender
    }
    example textual_audio:
    [00:00:00 - 00:00:05] Mary Taylor, however, related the tale of Zora to Mrs. Gray's private ear later.(Gender: Female, Emotion: neutral)
    """

    transcript = textual_audio.split("]")[-1].split("(")[0].strip()
    gender = textual_audio.split("Gender:")[-1].split(",")[0].strip()
    emotion = textual_audio.split("Emotion:")[-1].split(")")[0].strip()
    return {"ASR": transcript, "SER": emotion, "GR": gender}


def fill_other(item: dict) -> dict:
    labels = textual_audio_to_label(item["textual_audio"])
    if item["dataset"] == "Speech_emotion_recognition":
        return labels["SER"]
    elif item["dataset"] == "Gender_recognition":
        return labels["GR"]
    elif item["dataset"] == "Automatic_speech_recognition":
        return labels["ASR"]
    else:
        raise ValueError(f"Unknown dataset: {item['dataset']}")


def fill_mmau_answers(cot_data: List[dict]) -> List[dict]:
    id2data = {}
    data = json.load(open(MMAU_TEST_MINI_PATH))
    for item in data:
        id2data[item["id"]] = item

    for item in cot_data:
        labels = textual_audio_to_label(item["textual_audio"])
        if item["dataset"] == "MMAU":
            id = item["audio_filepath"].split("/")[-1].split(".")[0]
            if id not in id2data:
                raise ValueError(f"ID {id} not found in MMAU test mini data.")

            mmau_item = id2data[id]
            answer = mmau_item["answer"]
            item["label"] = answer
        elif item["dataset"] == "Speech_emotion_recognition":
            item["label"] = labels["SER"]
        elif item["dataset"] == "Gender_recognition":
            item["label"] = labels["GR"]
        elif item["dataset"] == "Automatic_speech_recognition":
            item["label"] = labels["ASR"]
        else:
            raise ValueError(f"Unknown dataset: {item['dataset']}")

    return cot_data


if __name__ == "__main__":
    with open(COT_PATH, "r") as f:
        cot_data = [json.loads(line) for line in f]

    filled_data = fill_mmau_answers(cot_data)

    with open(COT_PATH, "w") as f:
        for item in filled_data:
            f.write(json.dumps(item) + "\n")
