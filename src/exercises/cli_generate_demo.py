import json
from pathlib import Path
from src.exercises.generator import GenerateConfig, generate_from_payload

def main():
    # Demo payload similar to what you got
    payload = {
      "makharij_flags": [
        {"type":"seen_as_sad_to_seen","expected":"صراط","got":"سراط"},
        {"type":"merged_waw_la","expected":"ولا","got":"ولضالين"}
      ]
    }

    clips_map = {
      "ref_sirat": "artifacts/clips/ref_sirat.wav",
      "user_sirat": "artifacts/clips/user_sirat.wav",
      "ref_al_sirat": "artifacts/clips/ref_al_sirat.wav",
      "user_al_sirat": "artifacts/clips/user_al_sirat.wav",
      "ref_wala_daalin": "artifacts/clips/ref_wala_daalin.wav",
      "user_waddaalin": "artifacts/clips/user_waddaalin.wav",
    }

    timestamps_map = {
      "sirat_user": [26.00, 26.94],
      "sirat_ref": [53.20, 54.60],
      "waddaalin_user": [36.14, 37.54],
      "waddaalin_ref": [65.02, 69.00],
    }

    config = GenerateConfig()
    written = generate_from_payload(
        payload,
        config=config,
        user_full_wav="data/user_submissions/test_user/seg_full_0001/user_16k_vad.wav",
        clips_map=clips_map,
        timestamps_map=timestamps_map
    )

    print("WRITTEN:", len(written))
    for k,v in written.items():
        print("-", k, "->", v)

if __name__ == "__main__":
    main()
