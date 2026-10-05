# Frozen legacy-lcoa source: commit 94de8ce (3 May 2023)

Repository: https://github.com/cpalazzi/lcoa-opt.git, commit
`94de8ce` ("SHould have ben included in last push.", nsalmon11, 3 May 2023),
the last state committed before the shipping supplier tables were committed to
green-porpoise (2 June 2023) and before the 6 June 2023 commit (`cd56c11`)
that added the electrolysis water cost, the compressor marginal cost, the
hydrogen-store cycling constraint, the `time_step` store/marginal scaling, a
148,000 USD/MW/yr `BatteryInterfaceOut` cost and revised HB stoichiometry.
Its plant CSVs carry real (2030-vintage) costs that the workbook overrides
for the named equipment; `BatteryInterfaceOut` (0), `HydrogenCompression`
(1,000), `HydrogenFromStorage` (1) and the HB coefficients (6.27 / -6.97575)
are not overridden. Its `run_Alli_sites` solves the full 8,760-hour year.
Extracted with `git archive` on 15 September 2026; nothing edited.
`GeneralSteelData_20230831.xlsx` is the same workbook copy as in `cd56c11/`.
