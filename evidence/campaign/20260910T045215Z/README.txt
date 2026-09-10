CAMPAIGN 20260910T045215Z
{"verdict": "PASS", "bad_cases": 0}
TEXT_GATES 5/5
Traceback (most recent call last):
  File "<home>/glm53-flash-nvfp4/scripts/run_q200v2_glm53.py", line 133, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "<home>/glm53-flash-nvfp4/scripts/run_q200v2_glm53.py", line 85, in main
    summary = q.run_quality(
              ^^^^^^^^^^^^^^
  File "<home>/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner/scripts/run_quality_set.py", line 1583, in run_quality
    raise ValueError("image_id must be a full immutable sha256 image ID")
ValueError: image_id must be a full immutable sha256 image ID
START 25%
{"actual_depth": 0.25004371830422945, "api_prompt_tokens": 120136, "completion_tokens": 80, "elapsed_s": 152.805, "filler_token_width": 12, "finished_utc": "2026-09-10T04:59:33.609480+00:00", "label": "25%", "multikey": false, "needle_retrieved": true, "needle_start_tokens": 30027, "post_repeats": 8184, "pre_repeats": 2728, "raw_prompt_tokens": 120087, "requested_depth": 0.25, "response": "R0B0BENCH_NIAH_SLOT", "target_prompt_tokens": 131008, "transport_ok": true}
START 50%
{"actual_depth": 0.49987926426138873, "api_prompt_tokens": 120146, "completion_tokens": 76, "elapsed_s": 132.637, "filler_token_width": 12, "finished_utc": "2026-09-10T05:01:47.034145+00:00", "label": "50%", "multikey": false, "needle_retrieved": true, "needle_start_tokens": 60034, "post_repeats": 5457, "pre_repeats": 5456, "raw_prompt_tokens": 120097, "requested_depth": 0.5, "response": "R0B0BENCH_NIAH_SLOT", "target_prompt_tokens": 131008, "transport_ok": true}
START 90%
{"actual_depth": 0.8996735727133436, "api_prompt_tokens": 120137, "completion_tokens": 82, "elapsed_s": 126.515, "filler_token_width": 12, "finished_utc": "2026-09-10T05:03:54.502086+00:00", "label": "90%", "multikey": false, "needle_retrieved": true, "needle_start_tokens": 108040, "post_repeats": 1092, "pre_repeats": 9820, "raw_prompt_tokens": 120088, "requested_depth": 0.9, "response": "R0B0BENCH_NIAH_SLOT", "target_prompt_tokens": 131008, "transport_ok": true}
START mk33
{"api_prompt_tokens": 120151, "completion_tokens": 128, "distractor_nonces": ["ni4h7q2x9", "k7d2m8v4"], "elapsed_s": 150.373, "finished_utc": "2026-09-10T05:06:25.378423+00:00", "label": "mk33", "multikey": true, "needle_depths": [0.33, 0.66, 0.9], "needle_retrieved": true, "probe_nonce": "t3r9w5y1", "raw_prompt_tokens": 120102, "requested_depth": 0.33, "response": "t3r9w5y1", "target_prompt_tokens": 131008, "transport_ok": true}
START mk66
{"api_prompt_tokens": 120149, "completion_tokens": 127, "distractor_nonces": ["ni4h7q2x9", "k7d2m8v4"], "elapsed_s": 140.373, "finished_utc": "2026-09-10T05:08:46.250437+00:00", "label": "mk66", "multikey": true, "needle_depths": [0.33, 0.66, 0.9], "needle_retrieved": true, "probe_nonce": "t3r9w5y1", "raw_prompt_tokens": 120100, "requested_depth": 0.66, "response": "t3r9w5y1", "target_prompt_tokens": 131008, "transport_ok": true}
NIAH_PASS
CAMPAIGN_DONE 20260910T045215Z
