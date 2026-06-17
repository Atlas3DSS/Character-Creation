# Overnight Repo Snapshot

- Timestamp: 2026-04-04T23:29:48.631597
- Branch: `main`
- Modified tracked entries: 2
- Untracked entries: 124
- Control run completed: 9787 / 10240
- Control run remaining: 453
- Blackwell queue: `tmux` session `blackwell_reasoning_adapter`

## Notes
- This snapshot was written before sleep so the current repo/run drift is documented.
- The Blackwell queue is waiting for the dev-server reasoning-control run to finish, then it will export adapter data, continue trace/lean SFT, and benchmark.

## Git Status
```text
M scripts/infra/orchestrate_overnight.py
 M scripts/infra/queue_abliterated_connectome.sh
?? logs/baseline_activation_collection.log
?? logs/blackwell_reasoning_adapter_queue.log
?? logs/control_3090.log
?? logs/control_4090.log
?? logs/control_reasoning_v3_3090.log
?? logs/control_reasoning_v3_4090.log
?? logs/control_reasoning_v3_monitor.log
?? logs/control_v2_3090.log
?? logs/control_v2_4090.log
?? logs/download_glm47_local.log
?? logs/factorial_ablation.log
?? logs/nanochat_meta_think_compact_d8e3_v1.log
?? logs/nanochat_meta_think_compact_d8e3_watch.log
?? logs/nanochat_meta_think_compact_d8e3_watch.sh
?? logs/nanochat_meta_think_compact_pair_v1.log
?? logs/nanochat_meta_think_compact_pair_v2.log
?? logs/personality_meta_eval_v1_3090.log
?? logs/personality_meta_eval_v1_4090.log
?? logs/personality_meta_eval_v1_monitor.log
?? logs/personality_meta_eval_v1_monitor_runner.log
?? logs/personality_sweep_27b.log
?? logs/personality_sweep_blackwell.log
?? logs/queue_27b_after_baseline.log
?? logs/queue_27b_sweep.log
?? logs/queue_blocker.log
?? logs/repair_sampled25m_3090.log
?? logs/repair_sampled25m_4090.log
?? logs/sae_27b_train_L44_resume.log
?? logs/sweep_hourly_manager.log
?? logs/sweep_hourly_status.log
?? logs/sweep_remote_bootstrap.log
?? logs/sweep_v2_gpu0.log
?? logs/sweep_v2_gpu0a.log
?? logs/sweep_v2_gpu0b.log
?? logs/sweep_v2_gpu0c.log
?? logs/sweep_v2_workstation.log
?? logs/sweep_v2_ws4_a.log
?? logs/sweep_v2_ws4_b.log
?? logs/sweep_v2_ws4_c.log
?? logs/sweep_v2_ws4_d.log
?? logs/sweep_v2_ws4_e.log
?? logs/sweep_v2_ws_a.log
?? logs/sweep_v2_ws_b.log
?? logs/sweep_v2_ws_c.log
?? logs/sweep_v3_hourly_manager.log
?? logs/sweep_v3_ws_a.log
?? logs/sweep_v3_ws_b.log
?? logs/sweep_v3_ws_c.log
?? logs/trace_benchmark_compact_trace_d8_32_greedy.log
?? logs/ws15k_autohandoff.lock
?? logs/ws15k_autohandoff.log
?? logs/ws15k_fleet_3090.log
?? logs/ws15k_fleet_4090.log
?? logs/ws15k_fleet_bw.log
?? logs/ws15k_fleet_hourly.log
?? logs/ws15k_night_handoff.lock
?? logs/ws15k_night_handoff.log
?? logs/ws15k_pass2.log
?? logs/ws15k_pass2_resume_20260402.log
?? logs/ws15k_pass2_sampled25m_phase.log
?? logs/ws15k_repaired_responseonly_pass2.log
?? logs/ws15k_repaired_safe_analysis.log
?? logs/ws15k_repaired_safe_hourly.log
?? logs/ws15k_repaired_safe_overnight.log
?? logs/ws15k_repaired_safe_pass2_guard.log
?? logs/ws15k_sampled25m_analysis.log
?? logs/ws15k_server_restart.log
?? logs/ws15k_visualizer.log
?? logs/ws_fleet_3090.log
?? logs/ws_fleet_4090.log
?? logs/ws_fleet_bw.log
?? logs/ws_fleet_hourly.log
?? logs/ws_openai_hourly.log
?? logs/ws_openai_hourly_test.log
?? logs/ws_openai_pass1.log
?? reports/emotion_personality_experiment_plan_20260402.md
?? reports/qwen35_meta_think_probe_20260404.json
?? reports/qwen35_meta_think_prompting_notes_20260404.md
?? reports/ws15k_repaired_responseonly_phase_analysis_core/
?? reports/ws15k_repaired_responseonly_phase_visualizer.html
?? reports/ws15k_repaired_safe_handoff.md
?? reports/ws_openai_15k_sampled25m_phase_visualizer.html
?? restart_2026_03/
?? results/factorial_ablation/
?? scripts/experiments/personality/analyze_personality_phase_sweep.py
?? scripts/experiments/personality/benchmark_personality_trace_eval.py
?? scripts/experiments/personality/build_personality_visualizer.py
?? scripts/experiments/personality/export_nanochat_meta_think_compact.py
?? scripts/experiments/personality/export_nanochat_meta_think_probe.py
?? scripts/experiments/personality/export_reasoning_control_to_nanochat_adapter.py
?? scripts/experiments/personality/launch_personality_control_reasoning_reasoningonly_v3.sh
?? scripts/experiments/personality/launch_personality_meta_eval_v1.sh
?? scripts/experiments/personality/materialize_personality_meta_eval_condition.py
?? scripts/experiments/personality/monitor_pass1_loop.py
?? scripts/experiments/personality/monitor_pass1_progress.py
?? scripts/experiments/personality/monitor_personality_control_reasoning_reasoningonly_v3.sh
?? scripts/experiments/personality/monitor_personality_meta_eval_v1.sh
?? scripts/experiments/personality/monitor_repaired_safe_hourly.sh
?? scripts/experiments/personality/personality_control_reasoning_openai.py
?? scripts/experiments/personality/personality_meta_eval_openai.py
?? scripts/experiments/personality/personality_sweep_v2.py
?? scripts/experiments/personality/personality_sweep_v3_pass1_openai.py
?? scripts/experiments/personality/personality_sweep_v3_two_pass.py
?? scripts/experiments/personality/queue_blackwell_reasoning_adapter.sh
?? scripts/experiments/personality/repair_personality_generated_openai.py
?? scripts/experiments/personality/run_hourly_monitor_ws_openai.sh
?? scripts/experiments/personality/run_nanochat_meta_think_compact_pair.sh
?? scripts/experiments/personality/run_nanochat_meta_think_probe.sh
?? scripts/experiments/personality/run_personality_trace_benchmark.sh
?? scripts/experiments/personality/summarize_personality_meta_eval.py
?? scripts/experiments/personality/ws15k_autohandoff_pass2_bf16.py
?? scripts/experiments/personality/ws15k_night_handoff.sh
?? scripts/experiments/personality/ws15k_repaired_safe_overnight.sh
?? scripts/infra/launch_sweep_v2_hetero.sh
?? scripts/infra/launch_sweep_v3_hetero.sh
?? scripts/infra/remote_bootstrap_retry.sh
?? scripts/infra/run_with_vram_guard.py
?? scripts/infra/sweep_autotune.py
?? scripts/infra/sweep_status.py
?? scripts/infra/sweep_tps.py
?? sweep_v2/
?? sweep_v3/
?? sweep_v4/
?? ui/personality_phase_visualizer_template.html
```

## Blackwell Queue Tail
```text
[2026-04-04T23:20:08-07:00] waiting for control completion (360s elapsed)
[2026-04-04T23:21:08-07:00] waiting for control completion (420s elapsed)
[2026-04-04T23:22:08-07:00] waiting for control completion (480s elapsed)
[2026-04-04T23:23:08-07:00] waiting for control completion (540s elapsed)
[2026-04-04T23:24:08-07:00] waiting for control completion (600s elapsed)
[2026-04-04T23:25:08-07:00] waiting for control completion (660s elapsed)
[2026-04-04T23:26:08-07:00] waiting for control completion (720s elapsed)
[2026-04-04T23:27:08-07:00] waiting for control completion (780s elapsed)
[2026-04-04T23:28:08-07:00] waiting for control completion (840s elapsed)
[2026-04-04T23:29:08-07:00] waiting for control completion (900s elapsed)
```
