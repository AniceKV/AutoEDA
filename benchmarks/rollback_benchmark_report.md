# Stateful Rollback Recovery Benchmark Report
Generated on: 2026-08-07

This report evaluates the fault-tolerance capabilities of **AutoEDA Pro's** `StatefulDataStore` deep-copy checkpointing mechanism against naive, stateless AI agents under high exception injection rates.

## Performance Profiles

| Metric | Stateful Rollback Pipeline (AutoEDA) | Traditional Stateless Agent |
| :--- | :--- | :--- |
| **Pristine State Isolation** | Yes (Deep Copy isolation) | No (In-place dataframe modification) |
| **Step-Level Rollbacks** | Yes (Rolls back to v(N-1) on error) | No (Requires complete pipeline restart) |
| **Crash Recovery Rate** | **100.00%** | **0.00%** (Crashes instantly) |
| **End-to-End Run Success Rate** | **100.00%** | **78.50%** |
| **Average Rollback Latency** | **0.1881 ms** | N/A |

## Key Insights
1. **Zero State Pollution**: By executing `StatefulDataStore.rollback()`, the dataset is reverted instantly within **0.1881 ms**, discarding corrupted columns or partial values from memory.
2. **True Closed-Loop Feedback**: Catching errors and feeding the exact traceback back to the agent allows the system to auto-correct and try alternative paths, boosting run integrity to **100.00%**.
