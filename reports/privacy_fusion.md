# Privacy / fusion interface validation

Validation ID：`VAL-PRIVACY-FUSION-001`

Config hash：`4c289ee358f916d8e737d9d92c6589714b53dde74f9a12cd729ed57aa6f6742f`

Source split checksum：`ecba21a44e141523c692c3fbb092bb512b441de988408738f1ad5c9113562ea3`

## RUN validation

- privacy mode 傳入一個呼叫就會拋錯的 vision provider；provider 未被呼叫。
- 返回的 sensor probability 與輸入是同一 NumPy object，`tobytes()` 完全相同。
- fusion mode 沒有 provider 時明確拒絕，不會假裝已做 vision inference。
- aligned synthetic probabilities 可以固定 sensor weight `0.75` 執行 late fusion。
- 故意將 vision timestamps 全部平移 `1.0s`，會被 `0.25s` tolerance 拒絕。

這是 interface validation，不是模型效能 experiment。
`EXP-VISION-OF-LOSO-001` 與 `EXP-LATE-FUSION-LOSO-001` 的 registry status 都是
`NOT RUN`；本文沒有 vision-only 或 late-fusion performance 數字。
